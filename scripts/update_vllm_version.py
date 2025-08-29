# Standard
import argparse
import os
import sys
import tomllib

# Third Party
import tomli_w


def parse_args():

    help_str = "Updates vllm version on pyproject.toml"
    parser = argparse.ArgumentParser(description=help_str)
    parser.add_argument(
        "--vllm_version",
        dest="vllm_version",
        required=True,
        type=str,
        help="Version to overwrite pyproject.toml",
    )
    return parser.parse_args()


def parse_current_version(vllm_string):
    version = vllm_string.split("vllm.git@v", 1)[1].split(" ;")[0]
    return version


def main():

    args = parse_args()

    print(f"Trying to update vllm package version to {args.vllm_version}:")

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        # Find appropriate index
        vllm_deps = data["project"]["optional-dependencies"]["vllm"]
        index = next((i for i, item in enumerate(vllm_deps) if item.startswith("vllm @")), None)
        if not index:
            print(f"vllm dependency to be overwritten is missing - skipping update")
            sys.exit(0)
        current_version = parse_current_version(
            vllm_string=vllm_deps[index]
        )
        if current_version == args.vllm_version:
            print(
                f"VLLM already updated to latest version, skipping update and setting 'SKIP_VERSION' output!"
            )
            with open(os.getenv("GITHUB_OUTPUT"), "a") as env:
                print(f"SKIP_VERSION=true", file=env)
            sys.exit(0)
        else:
            data["project"]["optional-dependencies"]["vllm"][
                index
            ] = f"vllm @ git+https://github.com/vllm-project/vllm.git@v{args.vllm_version} ; sys_platform == 'darwin'"
            data["project"]["optional-dependencies"]["vllm"][
                1
            ] = f"vllm=={args.vllm_version} ; sys_platform != 'darwin'"

            with open("pyproject.toml", "wb") as f:
                tomli_w.dump(data, f)

            print(f"VLLM version updated to {args.vllm_version} successfully!")


if __name__ == "__main__":
    main()
