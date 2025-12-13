def normalize_python_code(code: str) -> str:
    """
    Convert literal \\n outside strings into real newlines,
    while preserving \\n inside quoted strings.
    """
    result = []
    in_single = False
    in_double = False
    i = 0

    while i < len(code):
        ch = code[i]

        if ch == "'" and not in_double:
            in_single = not in_single
            result.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
        elif ch == "\\" and not in_single and not in_double:
            # outside string: check for \n
            if i + 1 < len(code) and code[i + 1] == "n":
                result.append("\n")
                i += 1
            else:
                result.append(ch)
        else:
            result.append(ch)

        i += 1

    return "".join(result)


def run_python(code: str):
    import subprocess
    import sys
    import os
    import json
    from pathlib import Path

    print("Running python code...")

    # ✅ FIX: normalize only structural \n
    # code = normalize_python_code(code)

    base_dir = Path(__file__).parent
    code_dir = base_dir / "generated_code"
    code_dir.mkdir(exist_ok=True)

    file_path = code_dir / "solution_runner.py"

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Python code execution failed:\n{e.stderr.strip()}"
        ) from e

    output = result.stdout.strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output
