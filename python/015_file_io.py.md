# Concept: File I/O

## Concept ID

PYT-015

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Python Basics

## Learning Objectives

- Open files with open() using different modes (r, w, a, r+, w+, rb, wb)
- Use the with statement for automatic resource management
- Read files using ead(), eadline(), and eadlines()
- Write files using write() and writelines()
- Understand text vs binary file handling
- Use seek() and 	ell() for random access within files
- Work with the pathlib module for cross-platform path operations
- Handle file encoding properly
- Use the 	empfile module for temporary files

## Prerequisites

- Basic Python syntax (variables, functions, loops)
- Understanding of strings and byte data
- Error handling with try/except
- Context managers (with statement)

## Definition

File I/O (Input/Output) in Python refers to the process of reading data from files and writing data to files on disk. Python provides a rich set of tools for file operations, including the built-in open() function, file object methods, the with context manager for automatic cleanup, the pathlib module for modern path handling, and the 	empfile module for temporary file management. Python treats files as iterators, enabling memory-efficient processing of large files line by line.

## Intuition

Think of file handling like reading and writing in a physical notebook. open() is like picking up the notebook. The mode (, w, ) determines whether you are reading, writing from scratch (erasing everything first), or appending (adding to the end). The file object is your cursor position in the notebook. ead() is like reading everything from your current position to the end. seek() is like flipping to a specific page number. The with statement is like having a bookmark that automatically closes the notebook when you are done even if an exception occurs. pathlib is like having a GPS for your file system that works the same way on any operating system.

## Why This Concept Matters

Almost every non-trivial program interacts with the file system. You read configuration files, write logs, save user data, process datasets, and store results. In data science and AI/ML, file I/O is fundamental - you load training data from CSV or JSON files, save model checkpoints, write evaluation metrics, and read pretrained weights. Understanding file I/O ensures your programs can persist data between runs, handle large datasets efficiently, and work correctly across different operating systems. Proper file handling also prevents data corruption and resource leaks.

## Real World Examples

- **Configuration files**: Reading YAML, JSON, or INI configuration files at application startup
- **Log files**: Writing timestamped log entries for debugging and monitoring
- **Data export**: Exporting database query results to CSV or Excel files
- **Report generation**: Creating HTML, PDF, or text reports programmatically
- **File conversion**: Converting between file formats (CSV to JSON, Markdown to HTML)
- **Database storage**: SQLite databases are single files on disk
- **Web scraping**: Saving scraped HTML pages to disk for offline analysis

## AI/ML Relevance

- **Reading datasets**: Loading training data from CSV, JSON, Parquet, or text files
- **Saving model checkpoints**: Serializing model weights with pickle, joblib, or HDF5 during training
- **Logging training metrics**: Writing loss, accuracy, and other metrics to log files during training
- **Loading pretrained models**: Reading pretrained weights from files (e.g., PyTorch .pt, TensorFlow .h5)
- **Processing large text corpora**: Reading NLP datasets line by line to avoid memory overflow
- **Configuration management**: Reading YAML/JSON configs for experiment parameters
- **Writing predictions**: Saving model predictions to CSV or JSON files for submission or analysis

## Code Examples

### Example 1: Opening and Reading Files

`python
# Write a sample file first
with open("sample.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("This is line 2.\n")
    f.write("And this is line 3.\n")

# read() - entire file as a single string
with open("sample.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(repr(content))
# Output: 'Hello, World!\nThis is line 2.\nAnd this is line 3.\n'

# readline() - one line at a time
with open("sample.txt", "r", encoding="utf-8") as f:
    line1 = f.readline()
    line2 = f.readline()
    print(repr(line1), repr(line2))
# Output: 'Hello, World!\n' 'This is line 2.\n'

# readlines() - list of all lines
with open("sample.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(lines)
# Output: ['Hello, World!\n', 'This is line 2.\n', 'And this is line 3.\n']

# Iterating line by line (memory efficient)
with open("sample.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())
# Output:
# Hello, World!
# This is line 2.
# And this is line 3.
`

### Example 2: Writing Files

`python
# write() - writing a single string
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("First line\n")
    f.write("Second line\n")
    f.write("Third line\n")

# writelines() - writing a list of strings (no newlines added automatically)
lines = ["Apple\n", "Banana\n", "Cherry\n"]
with open("fruits.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)

# Verify
with open("fruits.txt") as f:
    print(f.read())
# Output:
# Apple
# Banana
# Cherry

# Append mode
with open("fruits.txt", "a", encoding="utf-8") as f:
    f.write("Date\n")

with open("fruits.txt") as f:
    print(f.read())
# Output:
# Apple
# Banana
# Cherry
# Date
`

### Example 3: File Modes

`python
# r - read (default)
with open("sample.txt", "r") as f:
    data = f.read()

# w - write (truncates existing file)
with open("new_file.txt", "w") as f:
    f.write("This overwrites any existing content.\n")

# a - append (adds to end, creates if missing)
with open("new_file.txt", "a") as f:
    f.write("This line is appended.\n")

# r+ - read and write (no truncation, starts at beginning)
with open("new_file.txt", "r+") as f:
    content = f.read()
    f.write("\n-- ADDED AT END --\n")

# w+ - read and write (truncates existing file)
with open("overwrite.txt", "w+") as f:
    f.write("Written with w+ mode.\n")
    f.seek(0)
    print(f.read())
# Output: Written with w+ mode.

# x - exclusive creation (raises FileExistsError if exists)
try:
    with open("new_file.txt", "x") as f:
        f.write("This will fail if file exists.\n")
except FileExistsError:
    print("File already exists!")
# Output: File already exists!
`

### Example 4: Binary Files

`python
# Writing binary data
with open("binary_data.bin", "wb") as f:
    f.write(b"Binary bytes: \x00\x01\x02\x03")
    f.write(bytes([10, 20, 30, 40, 50]))

# Reading binary data
with open("binary_data.bin", "rb") as f:
    data = f.read()
    print(data)
# Output: b'Binary bytes: \x00\x01\x02\x03\n\x14\x1e(2'

# Reading a binary file in chunks
with open("binary_data.bin", "rb") as f:
    while True:
        chunk = f.read(4)  # read 4 bytes at a time
        if not chunk:
            break
        print(f"Chunk: {chunk.hex()} -> {list(chunk)}")

# Copying a binary file
with open("binary_data.bin", "rb") as src:
    with open("binary_copy.bin", "wb") as dst:
        while chunk := src.read(4096):
            dst.write(chunk)
print("Binary copy complete.")
# Output: Binary copy complete.
`

### Example 5: seek() and tell()

`python
with open("sample.txt", "r") as f:
    print(f"Initial position: {f.tell()}")
    # Output: Initial position: 0

    f.read(6)  # read first 6 characters
    print(f"After reading 6 chars: {f.tell()}")
    # Output: After reading 6 chars: 6

    f.seek(0)  # go back to beginning
    print(f"After seek(0): {f.tell()}")
    # Output: After seek(0): 0

    f.seek(7)  # go to position 7 (0-indexed)
    print(f.read(5))
    # Output: World

    # seek with whence parameter
    f.seek(0, 2)  # seek from end - go to end
    print(f"End of file: {f.tell()}")
    # Output: End of file: 44

    f.seek(-6, 2)  # 6 bytes before end
    print(f.read())
    # Output: ne 3.
`

### Example 6: pathlib Module

`python
from pathlib import Path

# Creating Path objects
data_dir = Path("data")
config_file = Path("config/settings.json")

# Path operations
print(Path("sample.txt").exists())
# Output: True

print(Path("nonexistent.txt").exists())
# Output: False

print(Path("sample.txt").suffix)
# Output: .txt

print(Path("sample.txt").stem)
# Output: sample

# Creating directories
Path("output/logs").mkdir(parents=True, exist_ok=True)

# Writing and reading with pathlib
path = Path("output/hello.txt")
path.write_text("Hello from pathlib!\n", encoding="utf-8")
print(path.read_text(encoding="utf-8"))
# Output: Hello from pathlib!

# Listing files
for p in Path(".").glob("*.txt"):
    print(p.name)

# Recursive glob
for p in Path(".").glob("**/*.py"):
    print(p)
`

### Example 7: File Encoding

`python
# Writing with explicit encoding
with open("utf8_demo.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World! Python is fantastic!\n")

# Reading with the same encoding
with open("utf8_demo.txt", "r", encoding="utf-8") as f:
    print(f.read())
# Output: Hello, World! Python is fantastic!

# Reading with wrong encoding causes errors or garbled text
with open("utf8_demo.txt", "r", encoding="latin-1") as f:
    text = f.read()
    print(text[:30])

# Handling encoding errors
with open("utf8_demo.txt", "rb") as f:
    raw = f.read()

# Decode with error handling
print(raw.decode("utf-8", errors="replace")[:40])

print(raw.decode("latin-1", errors="replace")[:40])
`

### Example 8: tempfile Module

`python
import tempfile
from pathlib import Path

# Temporary file (automatically deleted when closed)
with tempfile.TemporaryFile(mode="w+") as tmp:
    tmp.write("Temporary data\n")
    tmp.seek(0)
    print(tmp.read())
# Output: Temporary data

# Named temporary file (visible in filesystem, auto-deleted)
with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=True) as tmp:
    print(f"Temp file path: {tmp.name}")
    tmp.write("Named temp file content\n")
    tmp.seek(0)
    print(tmp.read())

# Temporary directory
with tempfile.TemporaryDirectory() as tmp_dir:
    print(f"Temp dir: {tmp_dir}")
    temp_path = Path(tmp_dir) / "test.txt"
    temp_path.write_text("Inside temp dir\n")
    print(temp_path.read_text())
`

### Example 9: Reading Large Files Line by Line

`python
# Simulating a large file
with open("large_log.txt", "w", encoding="utf-8") as f:
    for i in range(100000):
        f.write(f"Line {i}: log entry number {i}\n")

# Memory-efficient reading - one line at a time
line_count = 0
with open("large_log.txt", "r", encoding="utf-8") as f:
    for line in f:  # reads one line at a time (not all into memory)
        line_count += 1
        if line_count <= 3:  # just show first 3
            print(line.strip())

print(f"Total lines: {line_count}")
# Output:
# Line 0: log entry number 0
# Line 1: log entry number 1
# Line 2: log entry number 2
# Total lines: 100000
`

## Common Mistakes

1. **Forgetting to close files**: Not using with leaves files open, causing resource leaks and potentially corrupted data. Always use with open(...) as f:.
2. **Opening files without specifying encoding**: Python uses platform-dependent default encoding. On Windows, this is usually cp1252 (not UTF-8), causing encoding errors when reading UTF-8 files created on other systems. Always specify encoding="utf-8".
3. **Assuming write() adds newlines**: .write("hello") does not add a newline. You must explicitly include \n. writelines() also does not add newlines.
4. **Reading entire large files into memory**: .read() loads the entire file into memory. For large files, iterate line by line with or line in f:.
5. **Using read() then write() without seeking**: After ead(), the file position is at the end. Writing without seek() will append, not overwrite from the beginning.
6. **Mixing text and binary modes**: Text mode expects strings; binary mode expects bytes. Passing the wrong type raises TypeError.
7. **Using os.path instead of pathlib.Path for path operations**: String-based path manipulation (os.path.join()) is error-prone on different platforms. pathlib is cleaner, safer, and cross-platform.
8. **Not handling FileNotFoundError**: File operations can fail. Always wrap file open calls in try/except when the file might not exist.
9. **Hardcoding path separators**: Using "folder/subfolder/file.txt" fails on Windows. Use pathlib or os.path.join() for cross-platform paths.
10. **Assuming tell() returns line numbers**: 	ell() returns byte position (in binary mode) or opaque cookie (in text mode), not line numbers.

## Interview Questions

### Beginner

1. **Q**: What does the with statement do when opening a file?
   **A**: The with statement creates a context manager that automatically calls .close() when the block exits, even if an exception occurs. This ensures the file is always properly closed, preventing resource leaks.

2. **Q**: What is the difference between ead(), eadline(), and eadlines()?
   **A**: ead(n) reads n characters (text) or n bytes (binary); with no argument, reads the entire file. eadline() reads one line (including newline). eadlines() reads all lines into a list. For large files, iterating or line in f: is most memory-efficient.

3. **Q**: What is the difference between 'w' and 'a' modes?
   **A**: 'w' (write) opens the file for writing, truncating (erasing) existing content first. 'a' (append) opens for writing, but preserves existing content and writes new data at the end.

4. **Q**: How do you read a file from a specific byte position?
   **A**: Use .seek(position) to move the file cursor to the specified byte position, then .read(n) to read from that position.

5. **Q**: What is the difference between text mode and binary mode?
   **A**: Text mode reads/writes strings and handles platform-specific newline conversion. Binary mode reads/writes bytes objects without any conversion.

### Intermediate

1. **Q**: How does Python handle newlines on different operating systems?
   **A**: In text mode, Python translates \n to the OS-specific line ending on write and reverse on read. In binary mode, no translation occurs. Use 
ewline='' parameter to disable translation.

2. **Q**: What is the purpose of 	empfile module?
   **A**: The 	empfile module provides functions for creating temporary files and directories that are automatically cleaned up. TemporaryFile() creates an unnamed temp file, NamedTemporaryFile() creates a visible temp file, and TemporaryDirectory() creates a temp directory.

3. **Q**: How would you efficiently read a 10GB log file line by line?
   **A**: Use or line in open("large.log", "r") which reads one line at a time without loading the entire file. The file object is an iterator that buffers reads internally (typically 8KB chunks) but yields individual lines.

4. **Q**: What is the difference between os.path and pathlib approaches?
   **A**: os.path uses string-based functions requiring manual string concatenation. pathlib.Path provides an object-oriented API with overloaded / operator, methods like .exists(), .read_text(), .glob(). pathlib is more readable and cross-platform.

5. **Q**: How do you handle encoding errors gracefully when reading a file?
   **A**: Specify encoding='utf-8' and handle UnicodeDecodeError in a try/except. Alternatively, use errors='replace' to replace undecodable bytes, or errors='ignore' to skip them.

### Advanced

1. **Q**: How does Python's file buffering work, and how can you control it?
   **A**: Python uses buffered I/O by default. For text files, the buffer is typically 8KB. Control buffering with the uffering parameter: uffering=0 (unbuffered), uffering=1 (line-buffered), or uffering=N (N-byte buffer).

2. **Q**: What is memory-mapped file I/O and when would you use it?
   **A**: Memory-mapped files (mmap module) map a file's contents directly into virtual memory, allowing you to access file data as if it were a bytearray. Use it for large files where you need random access without repeated system calls.

3. **Q**: How would you implement a thread-safe file writer that multiple threads can write to simultaneously?
   **A**: Use a 	hreading.Lock to serialize write access. Alternatively, have each thread write to its own buffer or temporary file, then merge after all threads complete.

## Practice Problems

### Easy

1. **File Line Counter**: Write a function that counts the number of lines in a text file.
2. **File Copier**: Write a function that copies a file from source to destination.
3. **Word Search**: Write a function that searches for a specific word in a file and returns the line numbers where it appears.
4. **Append Timestamp**: Write a function that appends the current timestamp to a log file.
5. **File Extension Check**: Write a function that checks if a given path has a specific extension.

### Medium

1. **CSV Reader**: Write a function that reads a CSV file and returns a list of dictionaries (header row as keys).
2. **Log Parser**: Write a function that parses a log file and extracts all ERROR-level log entries with timestamps.
3. **File Splitter**: Write a function that splits a large file into smaller files of N lines each.
4. **INI Parser**: Write a simple parser that reads a config file in INI format and returns a nested dictionary.
5. **JSON Lines Writer**: Write a function that reads a regular JSON array file and converts it to JSON Lines format.

### Hard

1. **Tail Implementation**: Implement the Unix 	ail command to read the last N lines of a file efficiently without reading the entire file.
2. **Directory Sync**: Write a function that synchronizes two directories, copying new/modified files from source to destination.
3. **Pattern-based File Merger**: Given a directory with multiple data files that share a common schema, write a function that merges them into a single output file.

## Solutions

### Easy Solutions

**1. File Line Counter**
`python
def count_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

with open("test_lines.txt", "w") as f:
    f.write("line1\nline2\nline3\n")
print(count_lines("test_lines.txt"))
# Output: 3
`

**2. File Copier**
`python
def copy_file(src, dst):
    with open(src, "rb") as source:
        with open(dst, "wb") as dest:
            while chunk := source.read(8192):
                dest.write(chunk)

copy_file("sample.txt", "sample_copy.txt")
print(Path("sample_copy.txt").exists())
# Output: True
`

**3. Word Search**
`python
def search_word(filename, word):
    results = []
    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if word.lower() in line.lower():
                results.append((line_num, line.strip()))
    return results

print(search_word("sample.txt", "line"))
# Output: [(1, 'Hello, World!'), (2, 'This is line 2.'), (3, 'And this is line 3.')]
`

**4. Append Timestamp**
`python
from datetime import datetime

def append_timestamp(filename):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Timestamp logged\n")

append_timestamp("timestamps.log")
`

**5. File Extension Check**
`python
from pathlib import Path

def has_extension(filepath, ext):
    return Path(filepath).suffix == ext

print(has_extension("sample.txt", ".txt"), has_extension("sample.txt", ".csv"))
# Output: True False
`

### Medium Solutions

**1. CSV Reader**
`python
def read_csv(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f]
        if not lines:
            return []
        headers = lines[0]
        return [dict(zip(headers, row)) for row in lines[1:]]
`

**2. Log Parser**
`python
def parse_error_logs(filename):
    errors = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if "ERROR" in line:
                parts = line.strip().split(" ", 2)
                if len(parts) >= 3:
                    errors.append({
                        "timestamp": parts[0],
                        "level": parts[1],
                        "message": parts[2]
                    })
    return errors
`

**3. File Splitter**
`python
def split_file(filename, lines_per_file=100):
    path = Path(filename)
    with open(filename, "r", encoding="utf-8") as f:
        part_num = 1
        while True:
            lines = []
            for _ in range(lines_per_file):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            part_filename = f"{path.stem}_part{part_num:03d}{path.suffix}"
            with open(part_filename, "w", encoding="utf-8") as part:
                part.writelines(lines)
            part_num += 1
`

**4. INI Parser**
`python
def parse_ini(filename):
    config = {}
    current_section = None
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                config[current_section] = {}
            elif "=" in line and current_section:
                key, _, value = line.partition("=")
                config[current_section][key.strip()] = value.strip()
    return config
`

**5. JSON Lines Writer**
`python
import json

def json_to_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
`

### Hard Solutions

**1. Tail Implementation**
`python
import os

def tail(filename, n=10):
    block_size = 8192
    with open(filename, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        lines_found = 0
        buffer = b""
        for offset in range(0, file_size, block_size):
            read_start = max(0, file_size - offset - block_size)
            read_size = min(block_size, file_size - read_start)
            f.seek(read_start)
            block = f.read(read_size)
            buffer = block + buffer
            lines_found = buffer.count(b"\n")
            if lines_found >= n + 1:
                break
        lines = buffer.decode("utf-8").splitlines()
        return lines[-n:]

with open("test_tail.txt", "w") as f:
    for i in range(100):
        f.write(f"Line {i}\n")
print(tail("test_tail.txt", 3))
# Output: ['Line 97', 'Line 98', 'Line 99']
`

**2. Directory Sync**
`python
import shutil
from pathlib import Path

def sync_dirs(src, dst, delete_extra=False):
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    for src_file in src_path.glob("**/*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(src_path)
            dst_file = dst_path / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {rel_path}")

    if delete_extra:
        for dst_file in dst_path.glob("**/*"):
            if dst_file.is_file():
                rel_path = dst_file.relative_to(dst_path)
                src_file = src_path / rel_path
                if not src_file.exists():
                    dst_file.unlink()
                    print(f"Deleted: {rel_path}")
`

**3. Pattern-based File Merger**
`python
import csv
import json
from pathlib import Path

def merge_files(input_dir, pattern, output_file, key_field=None):
    input_path = Path(input_dir)
    all_records = []
    seen_keys = set()

    for file_path in input_path.glob(pattern):
        if file_path.suffix == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if key_field:
                        k = row.get(key_field)
                        if k in seen_keys:
                            continue
                        seen_keys.add(k)
                    all_records.append(row)
        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for row in data if isinstance(data, list) else data.values():
                    if key_field:
                        k = row.get(key_field)
                        if k in seen_keys:
                            continue
                        seen_keys.add(k)
                    all_records.append(row)

    if all_records:
        ext = Path(output_file).suffix
        if ext == ".csv":
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
                writer.writeheader()
                writer.writerows(all_records)
        elif ext == ".json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2)
`

## Related Concepts

- Context managers (with statement)
- Error handling (try/except for file operations)
- String manipulation (formatting output)
- Serialization (pickle, json, csv modules)
- Binary data representation

## Next Concepts

- Working with CSV files using the csv module
- JSON serialization and deserialization
- Pickle for Python object serialization
- Database connectivity (sqlite3)
- Web APIs and HTTP requests (reading from URLs)

## Summary

File I/O in Python is handled through the built-in open() function, which returns a file object supporting various read and write operations. The with statement ensures automatic resource cleanup. Files can be opened in text or binary mode, with different access modes for reading, writing, and appending. The pathlib module provides a modern, object-oriented approach to filesystem path handling. The 	empfile module enables safe creation of temporary files. Understanding encoding, buffering, and efficient reading patterns is essential for building robust applications that handle data persistence correctly.

## Key Takeaways

- Use with open(...) as f: for automatic file closing
- Specify encoding="utf-8" for text files to avoid platform-dependent issues
- Modes:  (read), w (write/truncate),  (append), + (read/write), wb/b (binary)
- Read with ead(), eadline(), eadlines(), or iterate or line in f:
- Write with write() (single string) or writelines() (list of strings)
- Navigate with seek(position) and check position with 	ell()
- Use pathlib.Path for cross-platform path operations
- 	empfile.TemporaryFile() and 	empfile.TemporaryDirectory() for temporary storage
- Binary mode for non-text files (images, audio, serialized data)
- Handle FileNotFoundError, PermissionError, UnicodeDecodeError for robust code

