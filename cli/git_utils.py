import subprocess


class GitError(Exception):
    pass


def run_git(args: list[str], check: bool = False) -> tuple[str, str, int]:
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")

    return result.stdout, result.stderr, result.returncode


def is_git_repo() -> bool:
    _, _, code = run_git(["rev-parse", "--git-dir"])
    return code == 0


def get_repo_id() -> str:
    stdout, _, code = run_git(["remote", "get-url", "origin"])
    if code == 0 and stdout.strip():
        url = stdout.strip()
        url = url.replace("git@github.com:", "github.com/")
        url = url.replace("https://", "").replace(".git", "")
        return url

    stdout, _, _ = run_git(["rev-parse", "--show-toplevel"])
    if stdout.strip():
        return stdout.strip().split("/")[-1].split("\\")[-1]

    return "unknown-repo"


def get_staged_diff() -> str:
    stdout, _, _ = run_git(["diff", "--cached"])
    return stdout


def get_unstaged_diff() -> str:
    stdout, _, _ = run_git(["diff"])
    return stdout


def get_all_diff() -> str:
    stdout, _, _ = run_git(["diff", "HEAD"])
    return stdout


def create_commit(message: str) -> bool:
    _, _, code = run_git(["commit", "-m", message])
    return code == 0


def get_current_branch() -> str:
    stdout, _, _ = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return stdout.strip()


def get_tracked_files() -> list[str]:
    stdout, _, _ = run_git(["ls-files"])
    return [f for f in stdout.strip().split("\n") if f]


def stage_files(files: list[str]) -> bool:
    if not files:
        return True
    result = subprocess.run(
        ["git", "add"] + files,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitError(f"Failed to stage files: {result.stderr}")
    return True


def stage_tracked_changes() -> bool:
    _, _, code = run_git(["add", "-u"])
    return code == 0


def get_branch_diff(base_branch: str = "main") -> str:
    stdout, _, code = run_git(["diff", f"{base_branch}...HEAD"])
    if code != 0:
        stdout, _, _ = run_git(["diff", f"origin/{base_branch}...HEAD"])
    return stdout


def get_branch_commits(base_branch: str = "main") -> list[dict[str, str]]:
    fmt = "%H%x00%s%x00%b%x00%an"
    stdout, _, code = run_git([
        "log", f"{base_branch}..HEAD",
        f"--pretty=format:{fmt}",
        "--reverse",
    ])
    if code != 0 or not stdout.strip():
        return []

    commits = []
    for line in stdout.strip().split("\n"):
        parts = line.split("\x00")
        if len(parts) >= 4:
            commits.append({
                "hash": parts[0][:8],
                "subject": parts[1],
                "body": parts[2],
                "author": parts[3],
            })
    return commits


def unstage_all() -> bool:
    result = subprocess.run(
        ["git", "reset", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def get_merge_conflicts() -> list[str]:
    stdout, _, code = run_git(["diff", "--name-only", "--diff-filter=U"])
    if code != 0 or not stdout.strip():
        return []
    return [f for f in stdout.strip().split("\n") if f]


def get_conflict_content(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def get_ours_version(path: str) -> str:
    stdout, _, code = run_git(["show", f":2:{path}"])
    if code != 0:
        return ""
    return stdout


def get_theirs_version(path: str) -> str:
    stdout, _, code = run_git(["show", f":3:{path}"])
    if code != 0:
        return ""
    return stdout


def write_resolved_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_commits_between(from_ref: str, to_ref: str = "HEAD") -> list[dict[str, str]]:
    fmt = "%H%x00%s%x00%b%x00%an%x00%ai"
    stdout, _, code = run_git([
        "log", f"{from_ref}..{to_ref}",
        f"--pretty=format:{fmt}",
        "--reverse",
    ])
    if code != 0 or not stdout.strip():
        return []
    return _parse_log_output(stdout)


def get_commits_since(days: int) -> list[dict[str, str]]:
    fmt = "%H%x00%s%x00%b%x00%an%x00%ai"
    stdout, _, code = run_git([
        "log", f"--since={days}.days.ago",
        f"--pretty=format:{fmt}",
        "--reverse",
    ])
    if code != 0 or not stdout.strip():
        return []
    return _parse_log_output(stdout)


def get_tags() -> list[str]:
    stdout, _, code = run_git(["tag", "--sort=-version:refname"])
    if code != 0 or not stdout.strip():
        return []
    return [t for t in stdout.strip().split("\n") if t]


def _parse_log_output(stdout: str) -> list[dict[str, str]]:
    commits = []
    for line in stdout.strip().split("\n"):
        parts = line.split("\x00")
        if len(parts) >= 5:
            commits.append({
                "hash": parts[0][:8],
                "subject": parts[1],
                "body": parts[2],
                "author": parts[3],
                "date": parts[4].strip(),
            })
    return commits
