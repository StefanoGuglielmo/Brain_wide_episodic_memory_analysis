"""
setup_data.py

Populate the GitHub repository with the datasets downloaded from Zenodo.

Usage:
    python setup_data.py

Run this script from the root directory of the GitHub repository.
"""

from pathlib import Path
import shutil
import sys



#     "path_in_zenodo": "path_in_github_repository"

FILES = {
    "DATA/aggregated_colors.csv": "DATA/aggregated_colors.csv",
    "DATA/aggregated_mid.csv": "DATA/aggregated_mid.csv",
    "DATA/coarse.csv": "DATA/coarse.csv",
    "DATA/dictionary_for_aggregated.csv": "DATA/dictionary_for_aggregated.csv",
    "DATA/fine.csv": "DATA/fine.csv",
    "DATA/pls_bootstrap_ratio_CNTX_OPCRT.csv": "DATA/pls_bootstrap_ratio_CNTX_OPCRT.csv",
    "DATA/pls_bootstrap_ratio_HC_OPCRT.csv": "DATA/pls_bootstrap_ratio_HC_OPCRT.csv",
    "DATA/pls_bootstrap_ratio_OCT_OPCRT.csv": "DATA/pls_bootstrap_ratio_OCT_OPCRT.csv",
    "DATA/pls_dataset_CNTX_OPCRT.csv": "DATA/pls_dataset_CNTX_OPCRT.csv",
    "DATA/pls_dataset_HC_OPCRT.csv": "DATA/pls_dataset_HC_OPCRT.csv",
    "DATA/pls_dataset_OCT_OPCRT.csv": "DATA/pls_dataset_OCT_OPCRT.csv",
    "DATA/pls_significant_regions.csv": "DATA/pls_significant_regions.csv",

    "Fig_1/PCA_fine_level_df_standardized.csv":
        "Fig_1/PCA_fine_level_df_standardized.csv",

    "Fig_3/allen_connectivity_fine.csv":
        "Fig_3/allen_connectivity_fine.csv",

    "Fig_4/cntx_bootstrapped_metrics.csv":
        "Fig_4/cntx_bootstrapped_metrics.csv",
    "Fig_4/hub_cntx.csv":
        "Fig_4/hub_cntx.csv",
    "Fig_4/hub_oct.csv":
        "Fig_4/hub_oct.csv",
    "Fig_4/hub_opcrt.csv":
        "Fig_4/hub_opcrt.csv",
    "Fig_4/oct_bootstrapped_metrics.csv":
        "Fig_4/oct_bootstrapped_metrics.csv",
    "Fig_4/opcrt_bootstrapped_metrics.csv":
        "Fig_4/opcrt_bootstrapped_metrics.csv",
}



def print_header():
    print("\n" + "=" * 70)
    print("  DATA SETUP")
    print("=" * 70)
    print(
        "\nThis script copies the datasets downloaded from Zenodo into\n"
        "the appropriate folders of this GitHub repository.\n"
    )


def find_repository_root():
    """Return the directory containing this script."""
    return Path(__file__).resolve().parent


def ask_zenodo_directory():
    """Ask the user where the downloaded Zenodo data are located."""

    print(
        "Please provide the path to the folder containing the datasets\n"
        "downloaded from Zenodo.\n"
    )

    print("Example:")
    print("  C:\\Users\\YourName\\Downloads\\Zenodo_data")
    print()

    path_input = input("Zenodo data folder: ").strip().strip('"')

    if not path_input:
        print("\nNo folder was provided. Exiting.")
        sys.exit(1)

    zenodo_dir = Path(path_input).expanduser()

    if not zenodo_dir.exists():
        print(f"\nERROR: The folder does not exist:\n{zenodo_dir}")
        sys.exit(1)

    if not zenodo_dir.is_dir():
        print(f"\nERROR: The specified path is not a folder:\n{zenodo_dir}")
        sys.exit(1)

    return zenodo_dir


def check_files(zenodo_dir):
    """Check that all required files are present."""

    print("\nChecking Zenodo dataset...")

    missing = []

    for source in FILES:
        source_path = zenodo_dir / source

        if source_path.exists():
            print(f"  [OK]      {source}")
        else:
            print(f"  [MISSING] {source}")
            missing.append(source)

    if missing:
        print("\n" + "-" * 70)
        print("ERROR: Some required datasets are missing.")
        print("-" * 70)
        print(
            "\nPlease make sure that you downloaded the complete Zenodo\n"
            "dataset and selected the correct folder.\n"
        )

        print("Missing files:")
        for file in missing:
            print(f"  - {file}")

        print("\nNo files have been copied.")
        sys.exit(1)

    print("\nAll required datasets were found.")


def check_existing_files(repo_root):
    """Return destination files that already exist."""

    existing = []

    for destination in FILES.values():
        destination_path = repo_root / destination

        if destination_path.exists():
            existing.append(destination)

    return existing


def ask_overwrite(existing):
    """Ask whether existing files should be overwritten."""

    if not existing:
        return True

    print("\nThe following files already exist in the repository:")

    for file in existing:
        print(f"  - {file}")

    print()
    answer = input(
        "Do you want to overwrite these files? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


def copy_files(zenodo_dir, repo_root):
    """Copy all datasets to their destination."""

    print("\nCopying datasets...\n")

    for source, destination in FILES.items():

        source_path = zenodo_dir / source
        destination_path = repo_root / destination

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(source_path, destination_path)

        print(f"  [COPIED] {destination}")

    print("\nAll datasets have been copied successfully.")

def main():

    print_header()

    repo_root = find_repository_root()

    print(f"GitHub repository:\n{repo_root}\n")

    zenodo_dir = ask_zenodo_directory()

    print(f"\nZenodo dataset:\n{zenodo_dir}")

    check_files(zenodo_dir)

    existing = check_existing_files(repo_root)

    if not ask_overwrite(existing):
        print("\nNo files were copied. Exiting.")
        sys.exit(0)

    copy_files(zenodo_dir, repo_root)

    print("\n" + "=" * 70)
    print("  SETUP COMPLETE")
    print("=" * 70)
    print(
        "\nThe repository is now populated with the datasets required\n"
        "to run the analysis notebooks.\n"
    )

    print("You can now run the notebooks in the Fig_* folders.\n")


if __name__ == "__main__":
    main()