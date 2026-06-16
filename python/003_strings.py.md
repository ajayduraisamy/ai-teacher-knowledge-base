# Concept: Strings

## Concept ID

PYT-003

## Difficulty

BEGINNER

## Domain

Python

## Module

Python Basics

## Learning Objectives

After completing this lesson, you will be able to:

- Create strings using single quotes, double quotes, triple quotes, and raw strings
- Access individual characters using indexing and extract substrings using slicing
- Understand string immutability and its implications
- Use common string methods: `upper()`, `lower()`, `strip()`, `split()`, `join()`, `replace()`, `find()`, `count()`
- Format strings using f-strings and the `format()` method
- Use escape sequences for special characters
- Apply string concepts to real-world text processing tasks

## Prerequisites

- Understanding of variables and data types (PYT-001)
- Basic understanding of indexing and sequences

## Definition

A **string** in Python is an ordered, immutable sequence of characters. Strings are enclosed in quotes — single (`'`), double (`"`), triple single (`'''`), or triple double (`"""`). Every character in a string is a Unicode character, meaning Python strings support virtually every writing system in the world, from English to Chinese to emoji. Strings are **immutable**, which means once created, their content cannot be changed — any operation that appears to modify a string actually creates a new string.

## Intuition

Think of a string as a string of beads on a fixed thread. Each bead is a character, and you can look at any bead by its position (index), or examine a contiguous group of beads (slice). However, you cannot change an individual bead without cutting the thread and making a new string. When you "modify" a string in Python, you are always creating a new string object with the desired changes, just as you would make a new string of beads.

## Why This Concept Matters

Text data is everywhere in modern computing — user input, file contents, API responses, log files, web pages, and natural language for AI models. Python's string handling is one of its greatest strengths, offering a rich set of methods for text processing. Whether you are cleaning data, parsing CSV files, generating reports, building URLs, or preprocessing text for a machine learning model, you will work with strings constantly. Mastering strings is essential for almost every Python developer.

## Real World Examples

1. **Cleaning user input** — removing extra whitespace from a name entered in a form.
2. **Parsing log files** — splitting each line into timestamp, log level, and message.
3. **Building dynamic SQL queries** or API URLs using string formatting.
4. **Natural language processing (NLP)** — tokenizing sentences into words.
5. **Generating HTML or JSON** responses in a web application.

## AI/ML Relevance

Text data is central to natural language processing (NLP), one of the most active areas of AI/ML.

- **Tokenization** — splitting text into words or subword units using `split()` and more advanced tokenizers.
- **Text cleaning** — using `lower()`, `strip()`, `replace()` to normalize text before feeding it to a model.
- **Vocabulary building** — counting word frequencies with `count()` and collecting unique tokens.
- **Feature extraction** — converting strings into numerical representations like bag-of-words or TF-IDF.
- **Regular expressions** — pattern matching for complex text extraction (building on string methods).

```python
# AI/ML Example: Basic text preprocessing for NLP
raw_text = "   Hello World! This is an EXAMPLE text.   "

# Normalize
cleaned = raw_text.strip().lower()
print(cleaned)  # "hello world! this is an example text."

# Tokenize
tokens = cleaned.split()
print(tokens)  # ['hello', 'world!', 'this', 'is', 'an', 'example', 'text.']

# Count word occurrences
word_counts = {word: tokens.count(word) for word in set(tokens)}
print(word_counts)
# Output: {'hello': 1, 'is': 1, 'example': 1, 'this': 1, 'an': 1, 'world!': 1, 'text.': 1}
```

## Code Examples

### Example 1: Creating Strings

```python
# Single quotes
s1 = 'Hello'

# Double quotes
s2 = "World"

# Single and double quotes are interchangeable
# Use one style to embed the other
s3 = "It's a beautiful day"
s4 = 'He said "Python is great!"'

# Triple quotes for multi-line strings
s5 = """This is a
multi-line
string."""

print(s1)
print(s2)
print(s3)
print(s4)
print(s5)
# Output:
# Hello
# World
# It's a beautiful day
# He said "Python is great!"
# This is a
# multi-line
# string.
```

### Example 2: Indexing and Slicing

```python
text = "Python"

# Indexing (0-based)
print(text[0])     # P
print(text[3])     # h
print(text[-1])    # n (last character)
print(text[-2])    # o (second to last)

# Slicing: [start:stop:step]
print(text[0:3])   # Pyt (indices 0, 1, 2 — stop is exclusive)
print(text[:3])    # Pyt (start defaults to 0)
print(text[3:])    # hon (stop defaults to end)
print(text[::2])   # Pto (every other character)
print(text[::-1])  # nohtyP (reversed string)

# Important: out-of-range slices don't error
print(text[0:100])  # Python
```

### Example 3: String Immutability

```python
text = "Hello"
# text[0] = "J"  # TypeError: 'str' object does not support item assignment

# "Modifying" always creates a new string
new_text = "J" + text[1:]
print(new_text)   # Jello

# Reassignment changes which object the variable points to
text = text + " World"
print(text)       # Hello World
```

### Example 4: Common String Methods

```python
text = "  Python Programming  "

# Case conversion
print(text.upper())                # "  PYTHON PROGRAMMING  "
print(text.lower())                # "  python programming  "
print(text.capitalize())           # "  python programming  "
print(text.title())                # "  Python Programming  "
print(text.swapcase())             # "  pYTHON pROGRAMMING  "

# Whitespace removal
print(text.strip())                # "Python Programming"
print(text.lstrip())               # "Python Programming  "
print(text.rstrip())               # "  Python Programming"

# Searching
s = "Hello World"
print(s.find("World"))             # 6 (index where substring starts)
print(s.find("Python"))            # -1 (not found)
print(s.index("World"))            # 6 (like find, but raises ValueError if not found)
print(s.count("o"))                # 2

# Validation
print("abc123".isalnum())          # True
print("abc".isalpha())             # True
print("123".isdigit())             # True
print("   ".isspace())             # True
print("Hello".startswith("He"))    # True
print("Hello".endswith("lo"))      # True

# Replacement
print("I like cats".replace("cats", "dogs"))  # "I like dogs"
print("a,b,c".split(","))          # ['a', 'b', 'c']
print(" ".join(["a", "b", "c"]))   # "a b c"
```

### Example 5: String Formatting — f-Strings and format()

```python
name = "Alice"
age = 30
pi = 3.14159

# f-strings (Python 3.6+, preferred)
print(f"My name is {name} and I am {age} years old.")
# Output: My name is Alice and I am 30 years old.

# Expressions inside f-strings
print(f"Next year I will be {age + 1}.")
# Output: Next year I will be 31.

# Format specifiers
print(f"Pi to 2 decimals: {pi:.2f}")       # 3.14
print(f"Pi to 4 decimals: {pi:.4f}")       # 3.1416
print(f"Left aligned:    |{name:<10}|")     # |Alice     |
print(f"Right aligned:   |{name:>10}|")     # |     Alice|
print(f"Center aligned:  |{name:^10}|")     # |  Alice   |
print(f"Percentage: {0.25:.1%}")           # 25.0%

# format() method (older but still in use)
print("My name is {} and I am {} years old.".format(name, age))

# Named placeholders
print("My name is {name} and I am {age} years old.".format(name="Bob", age=25))
```

### Example 6: Escape Sequences and Raw Strings

```python
# Common escape sequences
print("Line1\nLine2")     # Newline: Line1 (newline) Line2
print("Tab\tseparated")   # Tab: Tab    separated
print("Backslash: \\")    # Backslash: \
print("Single quote: \'") # Single quote: '
print("Double quote: \"") # Double quote: "

# Unicode escape
print("\u0041")           # A (Unicode code point U+0041)
print("\U0001F600")       # 😀 (emoji)

# Raw strings — backslashes are treated literally
print(r"C:\Users\Name\Documents")   # C:\Users\Name\Documents
# Without raw: C:\Users\Name\Documents (would be interpreted)

# Raw strings are especially useful for regex patterns
import re
pattern = r"\d{3}-\d{4}"  # Raw string avoids double escaping
print(re.findall(pattern, "Call 555-1234 today!"))  # ['555-1234']
```

### Example 7: String Concatenation and Repetition

```python
# Concatenation with +
greeting = "Hello" + " " + "World"
print(greeting)           # Hello World

# Repetition with *
line = "-" * 20
print(line)               # --------------------

# Chaining with +=
msg = "Hello"
msg += " World"
print(msg)                # Hello World

# Join is more efficient for many strings
words = ["Python", "is", "awesome"]
sentence = " ".join(words)
print(sentence)           # Python is awesome

# Check membership
print("Py" in words[0])           # True
print("Java" not in words[0])     # True
```

## Common Mistakes

### 1. Trying to modify a string in place
```python
text = "Hello"
# text[0] = "J"  # TypeError

# Correct approach
text = "J" + text[1:]
```

### 2. Off-by-one errors in slicing
```python
text = "Python"
# text[0:6] gives "Python"
# text[0:5] gives "Pytho" (stop index is exclusive)
```

### 3. Using `+` for heavy string concatenation in loops
```python
# Inefficient: creates new strings each iteration
result = ""
for i in range(1000):
    result += str(i)  # O(n^2)

# Efficient: uses list and join
parts = []
for i in range(1000):
    parts.append(str(i))
result = "".join(parts)  # O(n)
```

### 4. Confusing `split()` and `join()`
```python
# split() turns a string into a list
# join() turns a list into a string
data = "a,b,c"
items = data.split(",")    # ['a', 'b', 'c']
back = ",".join(items)     # "a,b,c"
```

### 5. Forgetting that strings are immutable (methods return new strings)
```python
text = "hello"
text.upper()              # Returns "HELLO" but text is still "hello"
# Need to reassign:
text = text.upper()       # Now text is "HELLO"
```

### 6. Mixing types in string concatenation
```python
# TypeError: can only concatenate str (not "int") to str
# "Age: " + 25

# Correct
"Age: " + str(25)
```

### 7. Forgetting that `strip()` only removes leading/trailing characters
```python
# strip() does NOT remove internal whitespace
"Hello  World".strip()  # "Hello  World" (internal spaces remain)
```

## Interview Questions

### Beginner - 5

1. What is the difference between single quotes and double quotes for strings in Python?
2. How do you extract the last three characters of a string?
3. What does `s.split(",")` do? What type does it return?
4. How do you convert a string to lowercase?
5. What is the result of `"Python"[::-1]`?

### Intermediate - 5

1. Explain string immutability. How does it affect performance when modifying strings?
2. What is the difference between `find()` and `index()` string methods?
3. How do f-strings work? Show an example with formatting a float to 3 decimal places.
4. What is the difference between `str()` and `repr()`?
5. How does string interning work in Python? When are strings automatically interned?

### Advanced - 3

1. Explain how Python's `str.join()` works under the hood and why it is more efficient than string concatenation with `+` in loops.
2. What are raw strings and when would you use them? Provide a real-world regex example.
3. How does Python handle Unicode strings internally? Explain the difference between Python 2's `str` and `unicode` and Python 3's unified `str` type.

## Practice Problems

### Easy - 5 Questions

**Problem 1:** Create a string `s = "Hello, World!"` and slice it to get `"Hello"`.

**Problem 2:** Convert the string `"python is fun"` to title case and print it.

**Problem 3:** Count how many times the letter `"e"` appears in `"The quick brown fox jumps over the lazy dog"`.

**Problem 4:** Remove all leading and trailing spaces from `"   spaced out   "`.

**Problem 5:** Replace `"bad"` with `"good"` in the string `"This is a bad example"`.

### Medium - 5 Questions

**Problem 6:** Write a program that takes a user's first and last name, strips whitespace, capitalizes the first letter of each, and prints them as a full name.

**Problem 7:** Given `"apple,banana,orange,grape"`, split the string by commas, join the resulting list with `" | "`, and print the result.

**Problem 8:** Write a program that checks if a string is a palindrome (reads the same forwards and backwards). Ignore case and spaces.

**Problem 9:** Extract the domain name from the email `"user@example.com"` using `split()` and indexing.

**Problem 10:** Use f-strings to format a table header and two rows of data (name, age, score) with right-aligned columns of width 10.

### Hard - 3 Questions

**Problem 11:** Write a function `count_words(text)` that takes a string, strips punctuation, splits into words, and returns a dictionary of word frequencies.

**Problem 12:** Implement a Caesar cipher function `caesar(text, shift)` that shifts each letter by the given amount, wrapping around the alphabet.

**Problem 13:** Write a program that reads a multi-line string, finds all email addresses in it (strings containing `@` and `.`), and returns them as a list.

## Solutions

### Problem 1
```python
s = "Hello, World!"
print(s[:5])  # Hello
```

### Problem 2
```python
s = "python is fun"
print(s.title())  # Python Is Fun
```

### Problem 3
```python
text = "The quick brown fox jumps over the lazy dog"
print(text.count("e"))  # 3
```

### Problem 4
```python
text = "   spaced out   "
print(text.strip())  # "spaced out"
```

### Problem 5
```python
text = "This is a bad example"
print(text.replace("bad", "good"))  # "This is a good example"
```

### Problem 6
```python
first = "  alice  "
last = "  SMITH  "
full_name = first.strip().title() + " " + last.strip().title()
print(full_name)  # Alice Smith
```

### Problem 7
```python
fruits = "apple,banana,orange,grape"
fruit_list = fruits.split(",")
result = " | ".join(fruit_list)
print(result)  # apple | banana | orange | grape
```

### Problem 8
```python
def is_palindrome(s):
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]

print(is_palindrome("Racecar"))      # True
print(is_palindrome("A man a plan a canal Panama"))  # True
print(is_palindrome("Hello"))        # False
```

### Problem 9
```python
email = "user@example.com"
domain = email.split("@")[1]
print(domain)  # example.com
```

### Problem 10
```python
print(f"{'Name':>10} {'Age':>10} {'Score':>10}")
print(f"{'Alice':>10} {30:>10} {95.5:>10}")
print(f"{'Bob':>10} {25:>10} {88.0:>10}")
# Output:
#       Name        Age      Score
#      Alice         30       95.5
#        Bob         25       88.0
```

### Problem 11
```python
import string

def count_words(text):
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    words = text.lower().split()
    return {word: words.count(word) for word in set(words)}

sample = "Hello world! Hello everyone."
print(count_words(sample))
# Output: {'world': 1, 'everyone': 1, 'hello': 2}
```

### Problem 12
```python
def caesar(text, shift):
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26
            result.append(chr(base + shifted))
        else:
            result.append(char)
    return "".join(result)

print(caesar("Hello, World!", 3))  # Khoor, Zruog!
print(caesar("Khoor, Zruog!", -3)) # Hello, World!
```

### Problem 13
```python
import re

def find_emails(text):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)

sample = "Contact us at support@example.com or sales@company.org."
print(find_emails(sample))
# Output: ['support@example.com', 'sales@company.org']
```

## Related Concepts

- **Type Conversion** — Converting between strings and other types using `str()`, `int()`, `float()`
- **Booleans and Comparisons** — Checking string equality and ordering (lexicographic comparison)
- **Lists** — `split()` returns a list; `join()` operates on lists
- **Regular Expressions** — Advanced pattern matching built on string concepts
- **File I/O** — Reading from and writing to text files

## Next Concepts

- **Booleans and Comparisons** — Understanding truth values and comparisons
- **Type Conversion** — Converting safely between types
- **Lists** — Ordered, mutable sequences that complement strings

## Summary

Strings in Python are immutable sequences of Unicode characters created with single, double, or triple quotes. They support indexing (0-based), slicing (`[start:stop:step]`), and a rich set of methods for case conversion, whitespace handling, searching, replacement, splitting, and joining. F-strings provide a modern, readable way to embed expressions and format values within strings. Escape sequences handle special characters like newlines and tabs, while raw strings (prefixed with `r`) treat backslashes literally — essential for regular expressions and file paths. Understanding string methods and immutability is critical for effective text processing in any Python application.

## Key Takeaways

- Strings are immutable sequences of Unicode characters
- Create with `'`, `"`, `'''`, or `"""` quotes
- Index with `s[i]` (0-based); slice with `s[start:stop:step]`
- All "modifications" create new strings (immutability)
- Key methods: `upper()`, `lower()`, `strip()`, `split()`, `join()`, `replace()`, `find()`, `count()`
- Use f-strings (`f"...{expr}..."`) for formatting — they are concise and readable
- Raw strings (`r"..."`) keep backslashes literal
- Use `join()` for efficient concatenation of many strings
- Membership testing: `"sub" in string`
- String methods return new strings — remember to reassign!
