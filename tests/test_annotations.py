"""
Guards on the type annotations themselves: every function in the package
carries them, they agree with the defaults they are given, and no docstring
carries an invalid escape sequence.
"""

import ast
import pathlib
import warnings

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "dsptoolbox"


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
                    admitted = "bool" in parts
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
