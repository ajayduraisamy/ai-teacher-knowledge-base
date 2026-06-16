# Concept: Logging

## Concept ID

PYT-049

## Difficulty

Intermediate

## Domain

Python

## Module

Intermediate Python

## Learning Objectives

- Understand the logging module architecture (loggers, handlers, formatters, filters)
- Use the five standard logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Configure handlers: StreamHandler, FileHandler, RotatingFileHandler, TimedRotatingFileHandler
- Create and use custom formatters
- Organize loggers with `getLogger()` and hierarchical names
- Apply best practices for logging in production
- Implement logging in AI/ML training pipelines

## Prerequisites

- Basic Python functions and modules
- Understanding of `sys.stdout` and file I/O
- Basic configuration and environment variables

## Definition

The **`logging`** module is a flexible, standard-library framework for emitting log messages from applications. It provides a hierarchical logger system, configurable output destinations (handlers), customizable message formatting, and multiple severity levels. Logging is more sophisticated than `print()` because it supports categorization, filtering, routing, and persistent configuration.

## Intuition

Think of `print()` as a sticky note — useful for quick reminders, but messy and easy to lose. Logging is like a filing system — every message has a priority level, goes to the right folder (console, file, network), includes metadata (timestamp, source location), and can be turned on or off at different levels. You can have DEBUG messages during development and only WARNING+ in production without changing any code.

## Why This Concept Matters

Logging is critical for understanding, debugging, and monitoring software in production. Without logging, you are blind to what your application is doing. Proper logging enables troubleshooting issues without reproducing them, auditing security events, tracking performance, and understanding user behavior. The `logging` module's configurability means you can adapt it to any environment without changing application code.

## Real World Examples

- Web applications logging HTTP requests, errors, and response times
- Background task processors tracking job progress and failures
- CLI tools showing progress and verbose output
- Database migration scripts logging each step
- Microservices logging structured events for aggregation
- Security audit trails with access and authorization events

## AI/ML Relevance

Logging in AI/ML:
- Training loop logging (loss, accuracy, metrics per epoch)
- Data preprocessing pipeline logging (row counts, transformations)
- Model evaluation and validation logging
- Experiment tracking output for reproducibility
- Early stopping condition logging
- Resource utilization tracking (GPU memory, CPU usage)
- Error logging in production inference services

## Code Examples

### Example 1: Basic logging configuration

```python
import logging

# Basic configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

# Output:
# 2026-06-16 12:00:00 - root - DEBUG - This is a debug message
# 2026-06-16 12:00:00 - root - INFO - This is an info message
# 2026-06-16 12:00:00 - root - WARNING - This is a warning message
# 2026-06-16 12:00:00 - root - ERROR - This is an error message
# 2026-06-16 12:00:00 - root - CRITICAL - This is a critical message
```

### Example 2: Using getLogger with hierarchical names

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

# Create loggers with hierarchical names
app_logger = logging.getLogger('app')
app_logger.info('Application started')

db_logger = logging.getLogger('app.database')
db_logger.info('Database connection established')

query_logger = logging.getLogger('app.database.queries')
query_logger.debug('Executing SELECT * FROM users')

api_logger = logging.getLogger('app.api')
api_logger.warning('Slow response: 2.5s')

# Loggers inherit from parent; app.database.queries passes to app.database and app
# Output:
# app: Application started
# app.database: Database connection established
# app.api: Slow response: 2.5s
```

### Example 3: Custom handlers and formatters

```python
import logging
import sys

# Create logger
logger = logging.getLogger('custom')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('app.log', mode='a')
file_handler.setLevel(logging.DEBUG)

# Formatters
console_format = logging.Formatter('%(levelname)s: %(message)s')
file_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug('Only in file')
logger.info('Console and file')
logger.error('Error in both')

# Output (console):
# INFO: Console and file
# ERROR: Error in both
```

### Example 4: RotatingFileHandler

```python
import logging
from logging.handlers import RotatingFileHandler
import tempfile
import os

tmp_dir = tempfile.gettempdir()
log_path = os.path.join(tmp_dir, 'rotating.log')

logger = logging.getLogger('rotating')
logger.setLevel(logging.DEBUG)

# Rotate after 100 bytes, keep 3 backups
handler = RotatingFileHandler(
    log_path, maxBytes=100, backupCount=3
)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

for i in range(20):
    logger.info(f'Log entry number {i:03d}')

print(f'Generated files:')
for f in sorted(os.listdir(tmp_dir)):
    if f.startswith('rotating.log'):
        fpath = os.path.join(tmp_dir, f)
        size = os.path.getsize(fpath)
        print(f'  {f}: {size} bytes')

# Cleanup
for f in os.listdir(tmp_dir):
    if f.startswith('rotating.log'):
        os.remove(os.path.join(tmp_dir, f))

# Output:
# Generated files:
#   rotating.log: ... bytes
#   rotating.log.1: ... bytes
#   rotating.log.2: ... bytes
#   rotating.log.3: ... bytes
```

### Example 5: Structured logging with extra data

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] user=%(user)s %(message)s'
)

logger = logging.getLogger('app.auth')

def login(username, success):
    logger.info(
        'Login attempt',
        extra={'user': username}
    )
    if success:
        logger.info(
            'Login successful',
            extra={'user': username}
        )
    else:
        logger.warning(
            'Login failed',
            extra={'user': username}
        )

login('alice', True)
login('bob', False)

# Output:
# 2026-06-16 12:00:00 [INFO] user=alice Login attempt
# 2026-06-16 12:00:00 [INFO] user=alice Login successful
# 2026-06-16 12:00:00 [INFO] user=bob Login attempt
# 2026-06-16 12:00:00 [WARNING] user=bob Login failed
```

### Example 6: Exception logging with traceback

```python
import logging

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('app')

try:
    result = 10 / 0
except ZeroDivisionError:
    logger.exception('Division error occurred')

# Output:
# 2026-06-16 12:00:00 - ERROR - Division error occurred
# Traceback (most recent call last):
#   File "<stdin>", line 2, in <module>
# ZeroDivisionError: division by zero
```

### Example 7: AI/ML training logger

```python
import logging
import time
import random

class TrainingLogger:
    def __init__(self, log_file='training.log'):
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        self.logger.addHandler(console)
        self.logger.addHandler(fh)

    def start_training(self, config):
        self.logger.info('Training started')
        self.logger.info(f'Config: {config}')
        self.start_time = time.time()

    def log_epoch(self, epoch, metrics):
        elapsed = time.time() - self.start_time
        metrics_str = ', '.join(f'{k}={v:.4f}' for k, v in metrics.items())
        self.logger.info(f'Epoch {epoch:03d} - {metrics_str} - {elapsed:.1f}s')
        self.logger.debug(f'Epoch {epoch} details: {metrics}')

    def end_training(self, final_metrics):
        elapsed = time.time() - self.start_time
        self.logger.info(f'Training completed in {elapsed:.1f}s')
        self.logger.info(f'Final metrics: {final_metrics}')

config = {'lr': 0.001, 'batch_size': 32, 'epochs': 3}
train_logger = TrainingLogger()
train_logger.start_training(config)

for epoch in range(1, 4):
    metrics = {'loss': 0.5 / epoch, 'acc': 0.8 + 0.05 * epoch}
    time.sleep(0.05)
    train_logger.log_epoch(epoch, metrics)

train_logger.end_training({'val_loss': 0.12, 'val_acc': 0.95})

# Output:
# 2026-06-16 12:00:00 | training | INFO | Training started
# 2026-06-16 12:00:00 | training | INFO | Config: {'lr': 0.001, 'batch_size': 32, 'epochs': 3}
# 2026-06-16 12:00:00 | training | INFO | Epoch 001 - loss=0.5000, acc=0.8500 - 0.0s
# 2026-06-16 12:00:00 | training | INFO | Epoch 002 - loss=0.2500, acc=0.9000 - 0.1s
# 2026-06-16 12:00:00 | training | INFO | Epoch 003 - loss=0.1667, acc=0.9500 - 0.1s
# 2026-06-16 12:00:00 | training | INFO | Training completed in 0.2s
# 2026-06-16 12:00:00 | training | INFO | Final metrics: {'val_loss': 0.12, 'val_acc': 0.95}
```

### Example 8: Logger configuration via dictConfig

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'app.log',
            'maxBytes': 1024,
            'backupCount': 3,
        },
    },
    'loggers': {
        'app': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
        },
        'app.api': {
            'level': 'WARNING',
            'handlers': ['console'],
        },
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console'],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('app')
logger.info('Application configured via dictConfig')

# Output:
# INFO: Application configured via dictConfig
```

## Common Mistakes

1. Using `print()` for logging — loses all the benefits of levels, formatting, and routing
2. Creating multiple loggers in the same module with `logging.getLogger()` inside loops
3. Not using `if logger.isEnabledFor(logging.DEBUG)` for expensive log message construction
4. Forgetting that `logging.basicConfig()` only works if no handlers are already configured
5. Mixing different logging configurations that conflict with each other
6. Logging sensitive data (passwords, tokens, PII) at INFO or DEBUG level
7. Not handling uncaught exceptions with `logging.exception()` or `sys.excepthook`

## Interview Questions

### Beginner - 5

1. What are the five standard logging levels in Python?
2. What is the difference between `logging.info()` and `print()`?
3. How do you configure logging to write to a file?
4. What is a logger's effective level?
5. How do you change the format of log messages?

### Intermediate - 5

1. What is the logger hierarchy and how does it propagate messages?
2. How does `RotatingFileHandler` work and when would you use it?
3. What is the purpose of `logging.exception()` and how does it differ from `logging.error()`?
4. How do you add extra context to log messages with the `extra` parameter?
5. How does `dictConfig` enable centralized logging configuration?

### Advanced - 3

1. Design a logging system for a distributed microservices architecture with central aggregation.
2. How would you implement structured JSON logging for log aggregation tools (ELK, Splunk)?
3. How do you handle log buffering and asynchronous logging for high-throughput applications?

## Practice Problems

### Easy - 5

1. Configure logging to print messages to the console with a simple format.
2. Write a script that logs at all five levels and observe the output.
3. Create a logger that writes to a file named "app.log".
4. Add a timestamp to your log messages.
5. Change the logging level to WARNING and verify that DEBUG messages are suppressed.

### Medium - 5

1. Implement a logger with both console and file handlers at different levels.
2. Create a rotating file logger that logs to a new file daily.
3. Build a custom formatter that adds the function name and line number to each message.
4. Implement a logger that sends ERROR and CRITICAL messages via email.
5. Create a training log parser that reads a log file and extracts epoch metrics.

### Hard - 3

1. Design and implement a structured JSON logging system for an ML training pipeline.
2. Build a context-aware logger that tracks request IDs across asynchronous operations.
3. Implement a hierarchical logging system for a plugin-based application where each plugin has its own logger.

## Solutions

### Easy 1

```python
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info('Logging configured successfully')

# Output:
# INFO: Logging configured successfully
```

### Medium 1

```python
import logging

logger = logging.getLogger('multi_output')
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

file_h = logging.FileHandler('detail.log')
file_h.setLevel(logging.DEBUG)
file_h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(console)
logger.addHandler(file_h)

logger.debug('Debug to file only')
logger.info('Info to both')
logger.warning('Warning to both')

# Output (console):
# INFO: Info to both
# WARNING: Warning to both
```

### Hard 1

```python
import logging
import json

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger = logging.getLogger('structured')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info('Model training started', extra={'epoch': 1, 'loss': 0.5})

# Output:
# {"timestamp": "2026-06-16 12:00:00", "level": "INFO", "logger": "structured", "message": "Model training started"}
```

## Related Concepts

- The `print()` function and its limitations
- `warnings` module for non-fatal alerts
- `logging.handlers` module (RotatingFileHandler, SMTPHandler, SysLogHandler)
- Log aggregation tools (ELK Stack, Splunk, Datadog)
- Structured logging (JSON, logfmt)
- Observability and monitoring

## Next Concepts

- Unit testing (PYT-050)
- Configuration management
- Application monitoring
- Error tracking and reporting

## Summary

The `logging` module provides a powerful, flexible framework for emitting log messages from Python applications. With hierarchical loggers, multiple handlers, customizable formatters, and configurable levels, it supports everything from simple console output to complex enterprise logging infrastructure. Proper logging is essential for debugging, monitoring, and understanding application behavior.

## Key Takeaways

- Five levels: DEBUG, INFO, WARNING, ERROR, CRITICAL — control verbosity by level
- Use `getLogger(__name__)` for hierarchical, module-scoped loggers
- Handlers route log messages (console, file, network, etc.)
- Formatters control message layout and content
- `RotatingFileHandler` manages log file size and rotation
- `logging.exception()` logs with full traceback information
- Configure logging once, ideally via `dictConfig` or `basicConfig`
- AI/ML: log training progress, metrics, and configuration for reproducibility and debugging
