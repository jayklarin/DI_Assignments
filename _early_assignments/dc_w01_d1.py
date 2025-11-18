# Daily Challenge: Build Up A String

import random

# 1. Ask for user input
user_string = input("Enter a string of exactly 10 characters: ")

# 2. Check the length of the string
length = len(user_string)

if length < 10:
    print("String not long enough.")
elif length > 10:
    print("String too long.")
else:
    print("Perfect string!")

    # 3. Print the first and last characters
    print("First character:", user_string[0])
    print("Last character:", user_string[-1])

    # 4. Build the string character by character
    print("\nBuilding the string:")
    for i in range(1, len(user_string) + 1):
        print(user_string[:i])

    # 5. Bonus: Jumble the string
    print("\nBonus: Jumbled string:")
    char_list = list(user_string)
    random.shuffle(char_list)
    jumbled = "".join(char_list)
    print(jumbled)
