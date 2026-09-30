import random
import statistics


def generate_numbers(count=20, minimum=1, maximum=100):
    """
    Generate a list of random integers.
    """
    return [random.randint(minimum, maximum) for _ in range(count)]


def calculate_statistics(numbers):
    """
    Calculate basic statistics for a list of numbers.
    """
    return {
        "count": len(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
        "sum": sum(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "standard_deviation": (
            statistics.stdev(numbers)
            if len(numbers) > 1
            else 0
        ),
    }


def find_even_numbers(numbers):
    """
    Return only even numbers.
    """
    return [number for number in numbers if number % 2 == 0]


def find_odd_numbers(numbers):
    """
    Return only odd numbers.
    """
    return [number for number in numbers if number % 2 != 0]


def sort_numbers(numbers):
    """
    Return ascending and descending versions.
    """
    return {
        "ascending": sorted(numbers),
        "descending": sorted(numbers, reverse=True),
    }


def display_statistics(stats):
    print("\n--- Statistics ---")

    print(f"Count              : {stats['count']}")
    print(f"Minimum            : {stats['minimum']}")
    print(f"Maximum            : {stats['maximum']}")
    print(f"Sum                : {stats['sum']}")
    print(f"Mean               : {stats['mean']:.2f}")
    print(f"Median             : {stats['median']:.2f}")
    print(
        f"Standard Deviation : "
        f"{stats['standard_deviation']:.2f}"
    )


def display_menu():
    print("\n==============================")
    print(" Random Number Analysis Tool")
    print("==============================")
    print("1. Generate new numbers")
    print("2. Show statistics")
    print("3. Show even numbers")
    print("4. Show odd numbers")
    print("5. Sort numbers")
    print("6. Search for a number")
    print("7. Exit")


def search_number(numbers):
    try:
        target = int(input("Enter a number to search: "))

        positions = [
            index
            for index, value in enumerate(numbers)
            if value == target
        ]

        if positions:
            print(f"{target} was found.")
            print(f"Occurrences: {len(positions)}")
            print(f"Positions: {positions}")
        else:
            print(f"{target} was not found.")

    except ValueError:
        print("Please enter a valid integer.")


def main():
    numbers = generate_numbers()

    print("Random Number Analysis Program")
    print("Initial numbers:")
    print(numbers)

    while True:
        display_menu()

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            try:
                count = int(
                    input("How many numbers should be generated? ")
                )

                minimum = int(
                    input("Minimum value: ")
                )

                maximum = int(
                    input("Maximum value: ")
                )

                if count <= 0:
                    print("Count must be greater than zero.")
                    continue

                if minimum > maximum:
                    print(
                        "Minimum cannot be greater than maximum."
                    )
                    continue

                numbers = generate_numbers(
                    count,
                    minimum,
                    maximum
                )

                print("\nNew numbers:")
                print(numbers)

            except ValueError:
                print("Please enter valid integers.")

        elif choice == "2":
            stats = calculate_statistics(numbers)
            display_statistics(stats)

        elif choice == "3":
            even_numbers = find_even_numbers(numbers)

            print("\nEven numbers:")
            print(even_numbers)

            print(
                f"Total even numbers: "
                f"{len(even_numbers)}"
            )

        elif choice == "4":
            odd_numbers = find_odd_numbers(numbers)

            print("\nOdd numbers:")
            print(odd_numbers)

            print(
                f"Total odd numbers: "
                f"{len(odd_numbers)}"
            )

        elif choice == "5":
            sorted_values = sort_numbers(numbers)

            print("\nAscending:")
            print(sorted_values["ascending"])

            print("\nDescending:")
            print(sorted_values["descending"])

        elif choice == "6":
            search_number(numbers)

        elif choice == "7":
            print("\nProgram terminated.")
            break

        else:
            print(
                "Invalid option. "
                "Please select a number from 1 to 7."
            )


if __name__ == "__main__":
    main()
