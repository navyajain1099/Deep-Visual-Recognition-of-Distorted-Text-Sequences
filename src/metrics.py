def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def cer(predictions: list[str], targets: list[str]) -> float:
    total_distance = 0
    total_chars = 0
    for pred, target in zip(predictions, targets):
        total_distance += levenshtein(pred, target)
        total_chars += max(1, len(target))
    return total_distance / total_chars

