# Standard
from enum import Enum, auto
import argparse
import os

# Third Party
from vllm.utils.argparse_utils import FlexibleArgumentParser


class DetectorType(Enum):
    """Enum to represent different types of detectors"""

    TEXT_CONTENT = auto()
    TEXT_GENERATION = auto()
    TEXT_CHAT = auto()
    TEXT_CONTEXT_DOC = auto()


# This is taken from vLLM < 0.11.1 for backwards compatibility.
# vLLM versions >=0.11.1 no longer include StoreBoolean.
class StoreBoolean(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() == "true":
            setattr(namespace, self.dest, True)
        elif values.lower() == "false":
            setattr(namespace, self.dest, False)
        else:
            raise ValueError(
                f"Invalid boolean value: {values}. Expected 'true' or 'false'."
            )


# LocalEnvVarArgumentParser and dependent functions taken from
# https://github.com/opendatahub-io/vllm-tgis-adapter/blob/main/src/vllm_tgis_adapter/tgis_utils/args.py
# vllm by default parses args from CLI, not from env vars, but env var overrides
# may be useful for users. Here, we adopted the functionality inline
# to not add the vllm-tgis-adapter dependency by default, and in case the
# dependency becomes outdated


def _to_env_var(arg_name: str) -> str:
    return arg_name.upper().replace("-", "_")


def _bool_from_string(val: str) -> bool:
    return val.lower().strip() == "true" or val == "1"


def _switch_action_default(action: argparse.Action) -> None:
    """Switch to using env var fallback for all args."""
    env_val = os.environ.get(_to_env_var(action.dest))
    if not env_val:
        return

    val: bool | str
    # type=bool does not have the expected behavior, eg. bool("false") == True
    # Also handle special actions for boolean args
    if action.type is bool or type(action) in [
        argparse._StoreTrueAction,  # noqa: SLF001
        argparse._StoreFalseAction,  # noqa: SLF001
        StoreBoolean,
    ]:
        val = _bool_from_string(env_val)
    else:
        # for non-string args, the string value of the env var will be parsed
        # based on the action.type when setting from the default value
        val = env_val

    if action.nargs in ("+", "*"):
        action.default = [val]
    else:
        action.default = val


class LocalEnvVarArgumentParser(FlexibleArgumentParser):
    """Allows env var fallback for all args."""

    class _EnvVarHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
        def _get_help_string(self, action: argparse.Action) -> str:
            help_ = super()._get_help_string(action)
            assert help_ is not None

            if action.dest != "help":
                help_ += f" [env: {_to_env_var(action.dest)}]"
            return help_

    def __init__(
        self,
        parser: argparse.ArgumentParser | None = None,
        *,
        formatter_class: type[
            argparse.ArgumentDefaultsHelpFormatter
        ] = _EnvVarHelpFormatter,
        **kwargs,  # noqa: ANN003
    ):
        parents = []
        if parser:
            parents.append(parser)
            for action in parser._actions:  # noqa: SLF001
                if isinstance(action, argparse._HelpAction):  # noqa: SLF001
                    continue
                _switch_action_default(action)
        super().__init__(
            formatter_class=formatter_class, parents=parents, add_help=False, **kwargs
        )

    def _add_action(self, action: argparse.Action) -> argparse.Action:
        _switch_action_default(action)
        return super()._add_action(action)
