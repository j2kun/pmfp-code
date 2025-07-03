import sys

import numpy as np
from mpyc.runtime import mpc

# sample data that would come from parties on different servers
# columns are: ssn, age, score
provider0_data = np.array(
    [
        [123, 4, 1],
        [990, 3, 3],
        [111, 4, 5],
        [298, 3, 4],
    ],
    dtype=np.int16,
)

provider1_data = np.array(
    [
        [394, 2, 2],
        [983, 4, 2],
        [452, 2, 3],
        [399, 3, 5],
    ],
    dtype=np.int16,
)

provider2_data = np.array(
    [
        [987, 3, 3],
        [127, 3, 1],
        [100, 1, 2],
        [400, 1, 3],
    ],
    dtype=np.int16,
)


provider_data = [
    provider0_data,
    provider1_data,
    provider2_data,
]


async def main(provider_id: int):
    secint = mpc.SecInt(16)
    await mpc.start()

    # all_rows is a list of secret-shared arrays, one for each provider.
    #
    # The mpyc library works by having each process provides its data to
    # `mpc.input`, and the returned value is a secret-shared array of values
    # from all processes.
    all_rows = mpc.input(secint.array(provider_data[provider_id]))
    total = 0
    for provider in all_rows:
        for row in provider:
            # We can't use "and" for conditionals because the branch taken
            # would leak security. The MPC guarantee makes even trying this
            # impossible. Using multiplication and converting conditions to 0-1
            # valued integers avoids this.
            total += (row[1] >= 4) * (row[2] <= 2)
    print("Result:", await mpc.output(total))
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(main(int(sys.argv[1])))
