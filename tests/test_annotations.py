"""
Guards on the API surface itself: every function carries type annotations,
they agree with the defaults they are given, the documented parameters match
the signatures, no docstring carries an invalid escape sequence, and nothing
prints unless printing is what it is for.
"""

import ast
import pathlib
import re
import warnings

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "dsptoolbox"

# Their whole purpose is to write to the console
PRINTING_FUNCTIONS = ("show_info", "list_devices", "print_device_info")

DOCSTRING_SECTIONS = {
    "Parameters",
    "Returns",
    "Yields",
    "Notes",
    "References",
    "Examples",
    "Methods",
    "Attributes",
    "Attributes and Methods",
    "Raises",
    "See Also",
}


def _sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path.relative_to(PACKAGE.parent), path.read_text()


def _functions():
    for name, source in _sources():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield name, node


def _nullable_aliases() -> set[str]:
    """Names the package binds to a union that already contains None, so a
    None default under them is correct (e.g. `RngLike`).

    """
    aliases = set()
    for _, source in _sources():
        for node in ast.walk(ast.parse(source)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            if not targets or node.value is None:
                continue
            if "None" in ast.unparse(node.value):
                aliases |= {t.id for t in targets if isinstance(t, ast.Name)}
    return aliases


def _arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = node.args
    names = [
        a.arg
        for a in args.posonlyargs + args.args + args.kwonlyargs
        if a.arg not in ("self", "cls")
    ]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _documented_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """Names listed in the numpydoc `Parameters` section, or None when the
    docstring has no such section.

    """
    doc = ast.get_docstring(node)
    if not doc:
        return None
    lines = doc.splitlines()
    for start, line in enumerate(lines):
        if (
            line.strip() == "Parameters"
            and start + 1 < len(lines)
            and set(lines[start + 1].strip()) == {"-"}
        ):
            break
    else:
        return None

    names = []
    indent = None
    previous = ""
    for line in lines[start + 2 :]:
        if not line.strip():
            previous = line
            continue
        current = len(line) - len(line.lstrip())
        if indent is None:
            indent = current
        if current < indent:
            break
        # An underline right after a non-empty line starts the next section
        if set(line.strip()) == {"-"} and previous.strip():
            break
        if current == indent:
            if line.strip() in DOCSTRING_SECTIONS:
                break
            entry = re.match(r"^([A-Za-z_][A-Za-z0-9_, *]*?)\s*:\s", line.strip())
            if entry is not None:
                names += [n.strip() for n in entry.group(1).split(",")]
        previous = line
    return names


def _annotated_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield (argument, annotation, default) for every default that is a
    literal, so that it can be compared against its annotation.

    """
    positional = node.args.posonlyargs + node.args.args
    pairs = list(
        zip(
            positional[len(positional) - len(node.args.defaults) :],
            node.args.defaults,
            strict=True,
        )
    )
    pairs += [
        (arg, default)
        for arg, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=True
        )
        if default is not None
    ]
    for arg, default in pairs:
        if arg.annotation is not None and isinstance(default, ast.Constant):
            yield arg, ast.unparse(arg.annotation), default.value


class TestAnnotations:
    def test_every_function_has_a_return_annotation(self):
        missing = [
            f"{name}:{node.lineno} {node.name}"
            for name, node in _functions()
            if node.returns is None
        ]
        assert not missing, "Functions without a return annotation:\n" + "\n".join(
            missing
        )

    def test_every_argument_has_an_annotation(self):
        missing = []
        for name, node in _functions():
            args = node.args
            every = args.posonlyargs + args.args + args.kwonlyargs
            if args.vararg is not None:
                every.append(args.vararg)
            if args.kwarg is not None:
                every.append(args.kwarg)
            missing += [
                f"{name}:{node.lineno} {node.name}({arg.arg})"
                for arg in every
                if arg.annotation is None and arg.arg not in ("self", "cls")
            ]
        assert not missing, "Arguments without an annotation:\n" + "\n".join(missing)

    def test_documented_parameters_match_the_signature(self):
        """A renamed parameter whose docstring entry was not renamed with it
        documents an argument that does not exist, and leaves the real one
        undocumented. `center_frequenc_hz` survived that way in five
        beamformers.

        """
        wrong = []
        for name, node in _functions():
            documented = _documented_parameters(node)
            if documented is None:
                continue
            actual = _arguments(node)
            absent = [d for d in documented if d not in actual]
            undocumented = [a for a in actual if a not in documented]
            if absent or undocumented:
                wrong.append(
                    f"{name}:{node.lineno} {node.name} "
                    f"documented but absent: {absent}, "
                    f"present but undocumented: {undocumented}"
                )
        assert not wrong, (
            "Docstrings that disagree with their signature:\n" + "\n".join(wrong)
        )

    def test_nothing_prints_progress(self):
        """A library reports through return values and warnings. Only the
        few functions whose purpose is to write to the console may print.

        """
        offenders = []
        for name, source in _sources():
            tree = ast.parse(source)
            enclosing = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    for child in ast.walk(node):
                        enclosing.setdefault(child, node.name)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                    and enclosing.get(node) not in PRINTING_FUNCTIONS
                ):
                    offenders.append(f"{name}:{node.lineno} in {enclosing.get(node)}")
        assert not offenders, "Calls to print():\n" + "\n".join(offenders)

    def test_literal_defaults_agree_with_their_annotation(self):
        """A `None` or `False` default under an annotation that admits
        neither is how `spectrum_to_subtract: NDArray = False` slipped
        through: the default silently meant something the type did not.

        """
        opaque = {"Any", "object"} | _nullable_aliases()
        wrong = []
        for name, node in _functions():
            for arg, annotation, default in _annotated_defaults(node):
                parts = {p.strip(" \"'") for p in annotation.strip("\"'").split("|")}
                if parts & opaque:
                    continue
                if default is None:
                    admitted = "None" in parts
                elif isinstance(default, bool):
                    # A Literal admits exactly the values it enumerates
                    admitted = "bool" in parts or any(
                        p.startswith("Literal[") and repr(default) in p for p in parts
                    )
                else:
                    continue
                if not admitted:
                    wrong.append(
                        f"{name}:{node.lineno} {node.name}"
                        f"({arg.arg}: {annotation} = {default!r})"
                    )
        assert not wrong, (
            "Defaults that their annotation does not admit:\n" + "\n".join(wrong)
        )

    def test_no_invalid_escape_sequences(self):
        """A stray backslash in a docstring is a SyntaxWarning today and an
        error in a future Python.

        """
        offenders = []
        for name, source in _sources():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compile(source, str(name), "exec")
            offenders += [
                f"{name}:{w.lineno} {w.message}"
                for w in caught
                if issubclass(w.category, SyntaxWarning)
            ]
        assert not offenders, "Invalid escape sequences:\n" + "\n".join(offenders)
