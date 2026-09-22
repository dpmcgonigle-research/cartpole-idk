"""Click compatibility for the existing space-separated list options."""

import click


class VariadicOptionsCommand(click.Command):
    """Translate ``--option a b`` into repeated Click option occurrences."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Expand space-separated values for repeated options before Click parses them.

        Args:
            ctx: Active Click command context.
            args: Raw command-line argument tokens.
        """
        list_options = {
            flag
            for param in self.params
            if isinstance(param, click.Option) and param.multiple
            for flag in param.opts
        }
        normalized: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            flag = token.split("=", 1)[0]
            normalized.append(token)
            index += 1
            if flag not in list_options:
                continue
            if "=" not in token:
                if index >= len(args) or args[index].startswith("--"):
                    raise click.UsageError(f"Option {flag!r} requires an argument.", ctx)
                normalized.append(args[index])
                index += 1
            while index < len(args) and not args[index].startswith("--") and args[index] != "-h":
                normalized.extend([flag, args[index]])
                index += 1
        return super().parse_args(ctx, normalized)
