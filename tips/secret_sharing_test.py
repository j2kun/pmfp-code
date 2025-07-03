import subprocess


def test_secret_sharing():
    processes = [
        subprocess.Popen(
            ["python", "tips/secret_sharing.py", f"{i}", "-M3", f"-I{i}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(3)
    ]

    outputs = [p.communicate() for p in processes]
    outputs = [
        [x for x in stdout.split("\n") if "Result:" in x][0]
        for (stdout, stderr) in outputs
    ]
    assert len(outputs) == 3
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0] == "Result: 2"
