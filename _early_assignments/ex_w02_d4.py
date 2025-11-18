# Exercises XP

# Last Updated: May 21st, 2025

# 👩‍🏫 👩🏿‍🏫 What You’ll Learn

# File handling (reading files)
# Data structures (lists)
# Random number generation
# String manipulation
# JSON (parsing, modifying, and saving)


# 🌟 Exercise 1: Random Sentence Generator

# Goal: Create a program that generates a random sentence of a specified length
# from a word list.



# Key Python Topics:

# File handling (open(), read())
# Lists
# Random number generation (random.choice())
# String manipulation (split(), join(), lower())
# Error handling (try, except)
# Input validation


# Instructions:

# Download the provided word list and save it in your development directory.
# Create a function to read the words from the file.
# Create a function to generate a random sentence of a given length.
# Create a main function to handle user input and program flow.


# Step 1: Create the get_words_from_file function
# Create a function named get_words_from_file that takes the file path as an argument.
# Open the file in read mode ("r").
# Read the file content.
# Split the content into a list of words.
# Return the list of words.
import json
import os
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'words.txt')

def get_words_from_file(file_path=file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        my_str = f.read()
        words = my_str.split("\n")
        return words # list
# get_words_from_file()


# Step 2: Create the get_random_sentence function
# Create a function named get_random_sentence that takes
# the sentence length as an argument.
# Call get_words_from_file to get the list of words.
# Select a random word from the list length times.
# Create a sentence with the selected words.
# Convert the sentence to lowercase.
# Return the sentence.
def get_random_sentence(sentence_length):
    words = get_words_from_file()
    sentence_string = ""
    for c in range(sentence_length):
        sentence_string += f'{random.choice(words)} '
    return sentence_string
#print(get_random_sentence(6))


# Step 3: Create the main function

# Create a function named main.
# Print a message explaining the program’s purpose.
# Ask the user for the desired sentence length.
# Validate the user input:
# Check if it is an integer.
# Check if it is between 2 and 20 (inclusive).
# If the input is invalid, print an error message and exit.
# If the input is valid, call get_random_sentence with
# the length and print the generated sentence.
def main():
    print("This program prints a random sentence of N words from words.txt.")
    try:
        my_length = int(input("Enter an integer between 2 and 20 (inclusive): ").strip())
    except ValueError:
        print("Invalid input (not an integer). Exiting.")
        return   # stops the program

    if not (2 <= my_length <= 20):
        print("Number out of range. Exiting.")
        return   # stops the program

    sentence = get_random_sentence(my_length)
    print("Your random sentence:")
    print(sentence)

# if __name__ == "__main__":
#     main()


# 🌟 Exercise 2: Working With JSON

# Goal: Access a nested key in a JSON string, add a new key, and save the modified JSON to a file.



# Key Python Topics:

# JSON parsing (json.loads())
# JSON serialization (json.dump())
# Dictionaries
# File handling (open())


# Instructions:

# Using the follow code:

# import json
# sampleJson = """{
#    "company":{
#       "employee":{
#          "name":"emma",
#          "payable":{
#             "salary":7000,
#             "bonus":800
#          }
#       }
#    }
# }"


# Access the nested “salary” key.
# Add a new key “birth_date” wich value is of format “YYYY-MM-DD”, to the “employee” dictionary: "birth_date": "YYYY-MM-DD".
# Save the modified JSON to a file.


# Step 1: Load the JSON string

# Import the json module.
# Use json.loads() to parse the JSON string into a Python dictionary.


# Step 2: Access the nested “salary” key

# Access the “salary” key using nested dictionary access (e.g., data["company"]["employee"]["payable"]["salary"]).
# Print the value of the “salary” key.


# Step 3: Add the “birth_date” key

# Add a new key-value pair to the “employee” dictionary: "birth_date": "YYYY-MM-DD".
# Replace "YYYY-MM-DD" with an actual date.


# Step 4: Save the JSON to a file

# Open a file in write mode ("w").
# Use json.dump() to write the modified dictionary to the file in JSON format.
# Use the indent parameter to make the JSON file more readable.


# Step 1: Save the sample JSON string into a file
sampleJson = '''
{
   "company":{
      "employee":{
         "name":"emma",
         "payable":{
            "salary":7000,
            "bonus":800
         }
      }
   }
}
'''

import os

dir_path = os.path.dirname(os.path.realpath(__file__))  # folder where script is located

employee_file = os.path.join(dir_path, "employee.json")
employee_modified_file = os.path.join(dir_path, "employee_modified.json")

# Now use these variables for reading/writing
with open(employee_file, "w", encoding="utf-8") as f:
    f.write(sampleJson)

with open(employee_file, "r", encoding="utf-8") as f:
    data = json.load(f)

data["company"]["employee"]["birth_date"] = "1990-01-01"

with open(employee_modified_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)


print("Modified JSON saved to employee_modified.json")
