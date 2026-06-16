# Concept: Regular Expressions

## Concept ID

PYT-046

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand the purpose and syntax of regular expressions
- Use `re.match()`, `re.search()`, `re.findall()`, and `re.finditer()`
- Perform substitution with `re.sub()` and splitting with `re.split()`
- Define and use groups and named groups in patterns
- Employ character classes, quantifiers, anchors, and alternation
- Use flags: `IGNORECASE`, `MULTILINE`, `DOTALL`, `VERBOSE`
- Apply regex to text preprocessing in AI/ML pipelines

## Prerequisites

- Python strings and string methods
- Loops and basic iteration
- Understanding of escape characters in strings

## Definition

A **regular expression** (regex) is a sequence of characters that defines a search pattern. The `re` module provides functions for matching, searching, replacing, and splitting strings based on these patterns. Regular expressions enable powerful text processing that goes far beyond what basic string methods can achieve.

## Intuition

Think of a regular expression as a mini-programming language for describing text patterns. Instead of writing complex loops with many `if` conditions and string operations, you write a single pattern like `r'\b[A-Z][a-z]*\b'` to find all capitalized words. Regex patterns are concise, powerful, and portable across many programming languages.

## Why This Concept Matters

Regular expressions are fundamental to text processing, data cleaning, and validation tasks. They are used in virtually every programming language and are essential for tasks like extracting data from logs, validating email addresses, parsing configuration files, cleaning messy text data, and implementing search functionality. For AI/ML practitioners, regex is a core tool for text preprocessing.

## Real World Examples

- Validating email addresses, phone numbers, URLs
- Extracting IP addresses from server logs
- Parsing CSV, TSV, and other structured text formats
- Cleaning and normalizing text data for NLP pipelines
- Finding and replacing patterns in code refactoring
- Syntax highlighting in code editors

## AI/ML Relevance

Regular expressions in AI/ML:
- Text preprocessing for NLP: tokenization, normalization, cleaning
- Extracting features from raw text data
- Parsing structured fields from crawled web data
- Cleaning dataset metadata and annotations
- Validating input data formats before feeding to models
- Pattern-based data augmentation

## Code Examples

### Example 1: re.match and re.search

```python
import re

text = 'The quick brown fox jumps over the lazy dog'

# match: only matches at the beginning of the string
m = re.match(r'The', text)
print(m.group() if m else 'No match')

m = re.match(r'fox', text)
print(m.group() if m else 'No match')

# search: matches anywhere in the string
s = re.search(r'fox', text)
print(s.group())
print(s.start(), s.end())
print(text[s.start():s.end()])

# Output:
# The
# No match
# fox
# 16 19
# fox
```

### Example 2: re.findall and re.finditer

```python
import re

text = 'Emails: alice@example.com, bob@test.org, charlie@domain.co.uk'

# findall: returns all matches as a list
pattern = r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'
emails = re.findall(pattern, text)
print(emails)

# finditer: returns iterator of match objects
for match in re.finditer(pattern, text):
    print(f'{match.group()} at {match.start()}-{match.end()}')

# Output:
# ['alice@example.com', 'bob@test.org', 'charlie@domain.co.uk']
# alice@example.com at 8-26
# bob@test.org at 28-41
# charlie@domain.co.uk at 43-64
```

### Example 3: Groups and named groups

```python
import re

text = '2024-01-15 event: Python Conference'

# Named groups
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
m = re.search(pattern, text)
if m:
    print(m.group())
    print(m.group('year'))
    print(m.group('month'))
    print(m.group('day'))
    print(m.groups())

# Groups for extraction
log_line = 'ERROR 2024-01-15 10:30:45 Connection timeout on port 8080'
pattern = r'(?P<level>\w+)\s+(?P<timestamp>[\d\s:-]+)\s+(?P<message>.+)'
m = re.search(pattern, log_line)
if m:
    print(m.groupdict())

# Output:
# 2024-01-15
# 2024
# 01
# 15
# ('2024', '01', '15')
# {'level': 'ERROR', 'timestamp': '2024-01-15 10:30:45', 'message': 'Connection timeout on port 8080'}
```

### Example 4: re.sub for substitution

```python
import re

# Basic substitution
text = 'Hello, my name is Alice. I live in New York.'
result = re.sub(r'[AEIOUaeiou]', '*', text)
print(result)

# Substitution with groups
text = 'Date: 2024-01-15'
result = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1', text)
print(result)

# Named groups in substitution
result = re.sub(r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
                r'\g<day>/\g<month>/\g<year>', text)
print(result)

# Function as replacement
def uppercase_hex(match):
    return match.group(0).upper()

text = 'colors: #ff00aa, #00ff00, #0000ff'
result = re.sub(r'#[0-9a-fA-F]+', uppercase_hex, text)
print(result)

# Output:
# H*ll*, my n*m* *s *l*c*. I l*v* *n N*w Y*rk.
# Date: 15/01/2024
# Date: 15/01/2024
# colors: #FF00AA, #00FF00, #0000FF
```

### Example 5: re.split

```python
import re

# Split on multiple delimiters
text = 'apple, banana; cherry|date  fig'
result = re.split(r'[,;|\s]+', text)
print(result)

# Split with maxsplit
text = 'one,two,three,four,five'
result = re.split(r',', text, maxsplit=2)
print(result)

# Split while keeping delimiters (with group)
text = 'word1...word2...word3'
result = re.split(r'(\.\.\.)', text)
print(result)

# Output:
# ['apple', 'banana', 'cherry', 'date', 'fig']
# ['one', 'two', 'three,four,five']
# ['word1', '...', 'word2', '...', 'word3']
```

### Example 6: Character classes and quantifiers

```python
import re

text = 'My phone: 555-123-4567, office: 555-987-6543'

# Character classes
print(re.findall(r'\d+', text))
print(re.findall(r'\w+', text))
print(re.findall(r'\s', text))

# Quantifiers
print(re.findall(r'\d{3}-\d{3}-\d{4}', text))
print(re.findall(r'\d{3,4}', text))

# Greedy vs non-greedy
html = '<div><p>Hello</p><p>World</p></div>'
print(re.findall(r'<.*>', html))
print(re.findall(r'<.*?>', html))

# Output:
# ['555', '123', '4567', '555', '987', '6543']
# ['My', 'phone', '555', '123', '4567', 'office', '555', '987', '6543']
# [' ', ' ', ' ', ' ', ' ']
# ['555-123-4567', '555-987-6543']
# ['555', '123', '4567', '555', '987', '6543']
# ['<div><p>Hello</p><p>World</p></div>']
# ['<div>', '<p>', '</p>', '<p>', '</p>', '</div>']
```

### Example 7: Regex flags

```python
import re

text = """First Line
second line
Third Line"""

# IGNORECASE
print(re.findall(r'line', text, re.IGNORECASE))

# MULTILINE: ^ and $ match line boundaries
print(re.findall(r'^\w+', text))
print(re.findall(r'^\w+', text, re.MULTILINE))

# DOTALL: . matches newlines
text2 = 'Line 1\nLine 2\nLine 3'
print(re.findall(r'Line.*2', text2))
print(re.findall(r'Line.*2', text2, re.DOTALL))

# VERBOSE: readable patterns with comments
pattern = r'''
    \b          # word boundary
    \d{3}       # area code
    [-.]        # separator
    \d{3}       # prefix
    [-.]        # separator
    \d{4}       # line number
    \b          # word boundary
'''
print(re.findall(pattern, 'Call 555-123-4567 today!', re.VERBOSE))

# Output:
# ['Line', 'line', 'Line']
# ['First']
# ['First', 'second', 'Third']
# []
# ['Line 1\nLine 2']
# ['555-123-4567']
```

### Example 8: AI/ML text preprocessing

```python
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

def extract_features(text):
    return {
        'word_count': len(tokenize(text)),
        'sentence_count': len(re.findall(r'[.!?]+', text)),
        'has_numbers': bool(re.search(r'\d+', text)),
        'has_uppercase': bool(re.search(r'[A-Z]', text)),
        'emails': re.findall(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', text),
        'urls': re.findall(r'https?://\S+', text),
    }

sample = 'Contact us at support@example.com or visit https://example.com!'
print(clean_text(sample))
print(tokenize(sample))
print(extract_features(sample))

# Output:
# contact us at supportexamplecom or visit httpsexamplecom
# ['contact', 'us', 'at', 'support', 'example', 'com', 'or', 'visit', 'https', 'example', 'com']
# {'word_count': 11, 'sentence_count': 1, 'has_numbers': False, 'has_uppercase': True, 'emails': ['support@example.com'], 'urls': ['https://example.com']}
```

## Common Mistakes

1. Forgetting to use raw strings (`r'pattern'`) — backslashes are interpreted as escape sequences
2. Using `re.match()` when `re.search()` is needed — `match` only checks the start of the string
3. Greedy quantifiers causing over-matching — use `*?`, `+?`, `??` for non-greedy
4. Not escaping special characters (`.` matches any char, not just a period)
5. Overly complex patterns that are hard to read — use `VERBOSE` flag for complex patterns

## Interview Questions

### Beginner - 5

1. What is the difference between `re.match()` and `re.search()`?
2. What does the `.` character match in a regex pattern?
3. How do you make a pattern case-insensitive?
4. What is the difference between `*` and `+` quantifiers?
5. How do you capture a group in a regex?

### Intermediate - 5

1. What is the difference between `re.findall()` and `re.finditer()`?
2. How do named groups work and how do you access them?
3. What does the `re.VERBOSE` flag do and why is it useful?
4. How do you perform substitution using captured groups?
5. What is the difference between greedy and non-greedy matching?

### Advanced - 3

1. How would you use regex to parse a nested structure like balanced parentheses?
2. Explain backtracking in regex engines and how catastrophic backtracking occurs.
3. How would you optimize a regex pattern that processes large files?

## Practice Problems

### Easy - 5

1. Write a regex to find all digits in a string.
2. Validate if a string contains the word "python".
3. Replace all spaces with underscores in a string.
4. Extract all words that start with a capital letter.
5. Check if a string ends with ".py".

### Medium - 5

1. Validate email addresses with a regex pattern.
2. Extract all hashtags from a tweet (e.g., #python, #AI).
3. Parse and reformat dates from MM/DD/YYYY to YYYY-MM-DD.
4. Extract all URLs from an HTML document.
5. Split a sentence into words, preserving punctuation as separate tokens.

### Hard - 3

1. Write a regex-based lexer that tokenizes a simple programming language.
2. Build a regex pattern to match valid IPv4 addresses.
3. Implement a template engine that replaces `{{ variable }}` placeholders using regex.

## Solutions

### Easy 1

```python
import re
text = 'Room 42 on floor 3, building 7A'
print(re.findall(r'\d+', text))

# Output:
# ['42', '3', '7']
```

### Medium 1

```python
import re
def validate_email(email):
    pattern = r'^[\w.+-]+@[\w-]+\.[\w.-]+$'
    return bool(re.match(pattern, email))

print(validate_email('user@example.com'))
print(validate_email('invalid@.com'))

# Output:
# True
# False
```

### Hard 1

```python
import re

def tokenize_code(code):
    patterns = [
        ('NUMBER', r'\d+'),
        ('IDENTIFIER', r'[a-zA-Z_]\w*'),
        ('OPERATOR', r'[+\-*/=]'),
        ('PAREN', r'[()]'),
        ('WHITESPACE', r'\s+'),
    ]
    tokens = []
    pos = 0
    while pos < len(code):
        match = None
        for token_type, pattern in patterns:
            regex = re.compile(pattern)
            match = regex.match(code, pos)
            if match:
                if token_type != 'WHITESPACE':
                    tokens.append((token_type, match.group()))
                pos = match.end()
                break
        if not match:
            raise ValueError(f'Unexpected character at position {pos}')
    return tokens

code = 'x = 42 + y'
print(tokenize_code(code))

# Output:
# [('IDENTIFIER', 'x'), ('OPERATOR', '='), ('NUMBER', '42'), ('OPERATOR', '+'), ('IDENTIFIER', 'y')]
```

## Related Concepts

- Python string methods (`split`, `replace`, `find`, `startswith`, etc.)
- Raw strings in Python
- Unicode and locale handling in regex
- The `regex` third-party module (enhanced regex)
- Lexical analysis and parsing
- Natural language processing basics

## Next Concepts

- datetime module (PYT-047)
- JSON serialization (PYT-048)
- Logging (PYT-049)
- Text preprocessing for NLP

## Summary

Regular expressions provide a powerful, concise language for describing text patterns. The `re` module in Python offers functions for matching, searching, replacing, and splitting strings with regex patterns. While regex patterns can become complex, they are essential tools for text processing, data validation, and cleaning tasks in both general programming and AI/ML workflows.

## Key Takeaways

- Use raw strings (`r'pattern'`) to avoid escape sequence issues
- `re.match()` checks the start; `re.search()` checks anywhere; `re.findall()` finds all
- Groups `()` capture parts of a match; named groups `(?P<name>)` add semantic labels
- `re.sub()` replaces patterns; `re.split()` splits on patterns
- Flags: `IGNORECASE`, `MULTILINE`, `DOTALL`, `VERBOSE` modify matching behavior
- Non-greedy quantifiers (`*?`, `+?`) prevent over-matching
- AI/ML: regex is essential for text preprocessing, feature extraction, and data validation
