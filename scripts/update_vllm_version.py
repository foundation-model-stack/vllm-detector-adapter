import argparse
import json
import tomllib
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

def main():

    args = parse_args()

    print(
        f"Trying to update vllm package version to {args.vllm_version}:"
    )

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        data["project"]["optional-dependencies"]["vllm"][0] = (
            f"vllm @ git+https://github.com/vllm-project/vllm.git@v{args.vllm_version} ; sys_platform == 'darwin'"
        ) 
        data["project"]["optional-dependencies"]["vllm"][1] = (
            f"vllm=={args.vllm_version} ; sys_platform != 'darwin'"
        )

    with open("pyproject.toml", "wb") as f:
        tomli_w.dump(data, f)

    print(f"VLLM version updated to {args.vllm_version} successfully!")

if __name__ == "__main__":
    main()
