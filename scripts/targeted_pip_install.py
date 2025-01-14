"""A simple wrapper around pip that tries to handle duplicate installation of transitive dependencies
by allowing to install dependencies into a separate directory while transitive deps remain on default path.
"""

# Standard
from enum import Enum
from optparse import Values
from pprint import pprint
from typing import Dict, List, Optional
import argparse
import json
import os
import subprocess
import tempfile
import warnings

# Third Party
from pip._internal.metadata import get_default_environment
from pip._internal.req import InstallRequirement, constructors, req_file
from pip._internal.req.req_uninstall import UninstallPathSet
from pip._internal.utils.encoding import auto_decode


class RequirementType(Enum):
    requirement_file = 1
    package_path = 2

def get_pip_parsed_requirements(file_path):
    """This function uses pip's function to process the requirements file
    and replace the environment variable like GIT_TOKEN automatically
    in the url so that they can be processed later on (in-memory.
    This is required since the pip report doesn't contain these token
    information (rightly so), so we don't have other ways of getting to them
    """

    new_dep_req_parts = {}

    # Note: below code is mostly taken from pip._internal.req.req_file.py
    with open(file_path, "rb") as f:
        content = auto_decode(f.read())
        # below preprocess function will automatically replace the
        # environment variable
        lines_enum = req_file.preprocess(content)
        for line_number, line in lines_enum:
            args, options_str = req_file.break_args_options(line)
            constraints = False
            # TODO: Convert options_str to Values using `shlex.split(options_str)`
            opts = Values()
            parsed_line = req_file.ParsedLine(
                file_path, line_number, args, opts, constraints
            )
            parsed_req = req_file.handle_line(parsed_line)
            parts = constructors.parse_req_from_line(
                parsed_req.requirement, parsed_req.line_source
            )
            if parts.requirement:
                new_dep_req_parts[parts.requirement.name] = vars(parts.requirement)
            else:
                warnings.warn(f"parts.requirement empty. Values in parts {vars(parts)}")

    return new_dep_req_parts


def get_deps_install_list(
    requirements_filename_or_package_path=None,
    requirement_type=RequirementType.requirement_file,
    report_filename="/tmp/req_install_report.json",
    unknown_args: Optional[List[str]] = None,
):
    """Function to run dry-run for fetching the list of dependencies getting installed
    with given requirements_file or local package path and install that
    package along with its requirements in pyproject.toml or setup files.
    """
    dep_install_dict = {}

    if not unknown_args:
        unknown_args = []

    new_dep_req_parts = {}

    # NOTE: Platform information if specified by the library is already considered
    # in selection of wheel file when generating following report
    if requirement_type == RequirementType.requirement_file:
        # parse requirements_file into parts and store them in memory with
        # environment variables replaced. This will get used
        # in populating the dict returned from this function
        new_dep_req_parts = get_pip_parsed_requirements(
            requirements_filename_or_package_path
        )

        command = subprocess.run(
            [
                "pip",
                "install",
                "--dry-run",
                "-r",
                requirements_filename_or_package_path,
                "--report",
                report_filename,
                *unknown_args,
            ]
        )
    else:
        command = subprocess.run(
            [
                "pip",
                "install",
                "--dry-run",
                requirements_filename_or_package_path,
                "--report",
                report_filename,
                *unknown_args,
            ]
        )

    print("Command execution status: ", command)
    if command.returncode == 0:
        with open(report_filename, "r") as f:
            json_report = json.load(f)
            install_list = json_report.get("install", [])
            for dep in install_list:

                # Get the name of the dependency
                lib_name = dep.get("metadata", {}).get("name")
                if not lib_name:
                    raise "lib_name not found in pip report's metadata"

                if lib_name in new_dep_req_parts:
                    dep_install_dict[lib_name] = new_dep_req_parts[lib_name]
                else:
                    # This would be the case when requirements are provided without name
                    # for example: git+https://github.com/caikit/caikit-nlp.git@0.3.3
                    if "download_info" in dep and "vcs_info" in dep["download_info"]:
                        # TODO: Figure out better way to do below processing
                        # Check for vcs_info to check git installation
                        if dep["download_info"]["vcs_info"].get("vcs") == "git":
                            vcs_info = dep["download_info"]["vcs_info"]
                            assert (
                                "requested_version" not in vcs_info
                            ), "Dependency defined in incorrect format. Please check!"
                            vcs_info["url"] = dep["download_info"]["url"]
                            # Add details
                            dep_install_dict[lib_name] = dep["download_info"][
                                "vcs_info"
                            ]
                        else:
                            raise """vcs_info not present and dependency is not parsable \
                                by get_pip_parsed_requirements function"""
                    elif "download_info" in dep and "dir_info" in dep["download_info"]:
                        info = {}
                        lib_version = dep.get("metadata", {}).get("version")
                        extras = dep.get("requested_extras", [])
                        # Add directory as dep install info
                        info["version"] = lib_version
                        info["extras"] = extras  # would be a list
                        info["dir_info"] = dep["download_info"]["url"]
                        dep_install_dict[lib_name] = info
                    elif "version" in dep.get("metadata", {}):
                        # This is the case for example for transitive dependencies
                        # or things we are getting from report
                        # for lib getting installed from report, we can also
                        # get information like extras, is_direct etc
                        info = {}
                        lib_version = dep.get("metadata", {}).get("version")
                        extras = dep.get("requested_extras", [])
                        info["version"] = lib_version
                        info["extras"] = extras  # would be a list
                        dep_install_dict[lib_name] = info

    # Note: dep_install_dict contains, dict of either str:str or str:Dict,
    # where later on contains details about installation via git (vcs)
    return dep_install_dict

def get_already_installed_deps():
    """Function to get the list of dependencies to a file"""

    dep_list = []

    with tempfile.NamedTemporaryFile() as pip_temp_file:
        pip_list_filename = pip_temp_file.name
        # Save pip output to a file
        with open(pip_list_filename, "w") as f:
            command = subprocess.run(["pip", "list", "--format", "json"], stdout=f)

        if command.returncode == 0:
            with open(pip_list_filename, "r") as f:
                dep_list = json.load(f)
        else:
            # TODO: Handle error case
            dep_list = []

    # Reformat to be a dict format
    dep_dict = {}
    for dep in dep_list:
        dep_dict[dep["name"]] = dep["version"]
    return dep_dict

def check_conflict(installed_deps_dict, dep_install_dict):
    """Check if the installed deps are also referred in report as libraries to be installed"""

    duplicate_lib_dict = {}
    for dep_name, version in dep_install_dict.items():
        if dep_name in installed_deps_dict:
            # Assign version of already installed dep to duplicate list
            duplicate_lib_dict[dep_name] = installed_deps_dict[dep_name]

    return duplicate_lib_dict

def targeted_installation(
    dep_install_dict, target_folder_name, unknown_args: Optional[List[str]] = None
):
    """Function to install the requirements file into a specific location"""

    if not unknown_args:
        unknown_args = []

    # Check if folder exist, otherwise create
    target_path = os.path.abspath(target_folder_name)
    if not os.path.exists(target_path):
        print(f"Create {target_path} directory.")
        os.makedirs(target_path, exist_ok=False)

    dep_installation_list = []
    for lib_name, info in dep_install_dict.items():
        package = lib_name
        extras = info.get("extras")
        url = info.get("url")
        specifier = str(info.get("specifier", ""))
        version = str(info.get("version", ""))
        if "vcs" in info:
            # Direct vcs installation
            vcs = info["vcs"]
            requested_revision = info["requested_revision"]
            dep_installation_list.append(f"{vcs}+{url}@{requested_revision}")
        elif "dir_info" in info:
            # Install from directory
            url = info["dir_info"]
            # Note version tag is not used when installing from directory
            requested_revision = info["version"]
            dep_installation_list.append(f"{url}")
        else:
            if extras:
                # Following will create something like foo[a,b] where extras are set(a,b)
                extras = ",".join(extras)
                package += "[" + extras + "]"
            if url:
                # Add url to package
                package += "@" + url
            if specifier:  # specifier is not empty strings
                # Note specifier would contain things like ==, >, etc.
                package += specifier
            elif version:
                # `version`` is coming from reports which refers to exact version to be installed
                package += f"=={version}"

            # TODO: Add support for platform specific requirements / args / options
            dep_installation_list.append(package)

    # Installing individual libraries
    install_command = subprocess.run(
        [
            "pip",
            "install",
            # Since deps are already taken care of while generating list of deps to be installed
            # via get_deps_install_list function (it doesn't uses --no-dep)
            "--no-deps",
            "--target",
            target_path,
            *dep_installation_list,
            *unknown_args,
        ]
    )
    print("Command execution status: ", install_command)
    if not install_command.returncode == 0:
        exit(-1)


def delete_preinstalled_libs(dep_uninstall_dict: Dict):
    """Uninstall given libraries"""

    dep_list = []
    for lib_name, version in dep_uninstall_dict.items():
        dep_list.append(f"{lib_name}=={version}")

    # Pip uninstall doesn't support un-installation from a target
    # folder. Therefore, we are implementing our own version using underlying
    # un-installation functions. This is done by manually setting `dist.local`
    # to False

    auto_confirm = True
    verbose = True

    for dep_name in dep_list:
        print(f"Uninstalling dep name: {dep_name}")
        # NOTE: dep_name here contains the version as well
        req: InstallRequirement = constructors.install_req_from_line(dep_name)
        dist = get_default_environment().get_distribution(req.name)
        if not dist.local:
            # We have to override Distribution class
            # to take control of `local` property
            class CustomDistribution(type(dist)):
                def __init__(self, dist) -> None:
                    super().__init__(
                        dist._dist, dist._info_location, dist._installed_location
                    )

                @property
                def local(self):
                    return True

            # We have to override UninstallPathSet to override _permitted method
            # this is also required for uninstalling from non environmental
            # target folders
            class CustomUninstallPathSet(UninstallPathSet):
                def _permitted(self, _path):
                    return True

            dist = CustomDistribution(dist)
            uninstalled_pathset = CustomUninstallPathSet.from_dist(dist)
        else:
            # Not local distribution so we can proceed as usual
            uninstalled_pathset = UninstallPathSet.from_dist(dist)

        uninstalled_pathset.remove(auto_confirm, verbose)
        if uninstalled_pathset:
            uninstalled_pathset.commit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--requirement",
        required=False,
        default=None,
        type=str,
        help="Install from the given requirements file. Note: this is taking over pip's -r",
    )
    parser.add_argument(
        "-p",
        "--path",
        required=False,
        default=None,
        type=str,
        help="Install a package from path.",
    )
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        default=None,
        type=str,
        help="Install packages into a specific directory. Note: This is taking over pip's -t",
    )
    parser.add_argument(
        "-di",
        "--delete_installed_deps",
        default=False,
        action="store_true",
        help="Uninstall already installed dependencies that are getting overridden in this installation. Note: Deletion will happen as last step",
    )

    args, unknown_args = parser.parse_known_args()

    # Make sure one of -r or -p is provided.
    if not args.path and not args.requirement:
        print(
            "One of -p or -r should be provided to install the new package / requirements"
        )
        exit(1)
    elif args.path and args.requirement:
        print("Only one of -p and -r can be provided not both")
        exit(1)

    if args.path:
        requirement_type = RequirementType.package_path
    else:
        requirement_type = RequirementType.requirement_file

    ## TODO: Add support for installing directly, i.e outside of requirements file
    requirements_file_or_path = args.requirement or args.path
    target_installation_dir = args.target

    installed_deps_dict = get_already_installed_deps()
    dep_install_dict = get_deps_install_list(
        requirements_filename_or_package_path=requirements_file_or_path,
        requirement_type=requirement_type,
        unknown_args=unknown_args,
    )

    # Check conflict
    duplicate_lib_dict = check_conflict(installed_deps_dict, dep_install_dict)

    if len(duplicate_lib_dict) != 0:
        warnings.warn(
            "Duplicate library installation detected for following libraries: "
        )
        pprint(duplicate_lib_dict, indent=4)
    else:
        print("No lib duplication detected")

    targeted_installation(
        dep_install_dict, target_installation_dir, unknown_args=unknown_args
    )

    if args.delete_installed_deps:
        print("Deleting pre-installed duplicate libraries as requested")
        delete_preinstalled_libs(duplicate_lib_dict)

