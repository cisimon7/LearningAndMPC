from multiprocessing import Pool, Process


def f(x):
    return x*x


if __name__ == '__main__':
    with Pool(5) as p:
        results = p.map(f, [1, 2, 3])

    print(results)
