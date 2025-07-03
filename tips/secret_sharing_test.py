import subprocess

from tips.secret_sharing import provider_data


def expected():
    count = 0
    for data in provider_data:
        for row in data:
            if row[1] >= 4 and row[2] <= 2:
                count += 1
    return count


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
    try:
        outputs = [
            [x for x in stdout.split("\n") if "Result:" in x][0]
            for (stdout, stderr) in outputs
        ]
    except Exception:
        print(outputs[0][0])
        print(outputs[0][1])
        assert False, "Error in subprocess execution"

    assert len(outputs) == 3
    assert outputs[0] == outputs[1] == outputs[2]

    actual = int(outputs[0].split(":")[1].strip())
    assert actual == expected()
