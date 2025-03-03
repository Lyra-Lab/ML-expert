import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Tuple, Optional

from fuzzysearch import find_near_matches
from rapidfuzz import fuzz


class TestType(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    ORDER = "order"

class TestChecked(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


@dataclass(kw_only=True)
class BasePDFTest:
    """
    Base class for all PDF test types.
    
    Attributes:
        pdf: The PDF filename.
        page: The page number for the test.
        id: Unique identifier for the test.
        type: The type of test.
        threshold: A float between 0 and 1 representing the threshold for fuzzy matching.
    """
    pdf: str
    page: int
    id: str
    type: str
    threshold: float = 1.0
    checked: Optional[TestChecked] = None

    def __post_init__(self):
        self.threshold = float(self.threshold)
        
        if not self.pdf:
            raise ValidationError("PDF filename cannot be empty")
        if not self.id:
            raise ValidationError("Test ID cannot be empty")
        if not isinstance(self.threshold, float) or not (0 <= self.threshold <= 1):
            raise ValidationError(f"Threshold must be a float between 0 and 1, got {self.threshold}")
        if self.type not in {t.value for t in TestType}:
            raise ValidationError(f"Invalid test type: {self.type}")

    def run(self, md_content: str) -> Tuple[bool, str]:
        """
        Run the test on the provided markdown content.
        
        Args:
            md_content: The content of the .md file.
            
        Returns:
            A tuple (passed, explanation) where 'passed' is True if the test passes,
            and 'explanation' provides details when the test fails.
        """
        raise NotImplementedError("Subclasses must implement the run method")


@dataclass
class TextPresenceTest(BasePDFTest):
    """
    Test to verify the presence or absence of specific text in a PDF.
    
    Attributes:
        text: The text string to search for.
    """
    text: str

    def __post_init__(self):
        super().__post_init__()
        if self.type not in {TestType.PRESENT.value, TestType.ABSENT.value}:
            raise ValidationError(f"Invalid type for TextPresenceTest: {self.type}")
        if not self.text.strip():
            raise ValidationError("Text field cannot be empty")

    def run(self, md_content: str) -> Tuple[bool, str]:
        reference_query = self.text
        threshold = self.threshold
        best_ratio = fuzz.partial_ratio(reference_query, md_content) / 100.0

        if self.type == TestType.PRESENT.value:
            if best_ratio >= threshold:
                return True, ""
            else:
                msg = (
                    f"Expected '{reference_query[:40]}...' with threshold {threshold} "
                    f"but best match ratio was {best_ratio:.3f}"
                )
                return False, msg
        else:  # ABSENT
            if best_ratio < threshold:
                return True, ""
            else:
                msg = (
                    f"Expected absence of '{reference_query[:40]}...' with threshold {threshold} "
                    f"but best match ratio was {best_ratio:.3f}"
                )
                return False, msg


@dataclass
class TextOrderTest(BasePDFTest):
    """
    Test to verify that one text appears before another in a PDF.
    
    Attributes:
        before: The text expected to appear first.
        after: The text expected to appear after the 'before' text.
    """
    before: str
    after: str

    def __post_init__(self):
        super().__post_init__()
        if self.type != TestType.ORDER.value:
            raise ValidationError(f"Invalid type for TextOrderTest: {self.type}")
        if not self.before.strip():
            raise ValidationError("Before field cannot be empty")
        if not self.after.strip():
            raise ValidationError("After field cannot be empty")

    def run(self, md_content: str) -> Tuple[bool, str]:
        threshold = self.threshold
        max_l_dist = round((1.0 - threshold) * len(self.before))
        before_matches = find_near_matches(self.before, md_content, max_l_dist=max_l_dist)
        after_matches = find_near_matches(self.after, md_content, max_l_dist=max_l_dist)

        if not before_matches:
            return False, f"'before' text '{self.before[:40]}...' not found with max_l_dist {max_l_dist}"
        if not after_matches:
            return False, f"'after' text '{self.after[:40]}...' not found with max_l_dist {max_l_dist}"

        for before_match in before_matches:
            for after_match in after_matches:
                if before_match.start < after_match.start:
                    return True, ""
        return False, (
            f"Could not find a location where '{self.before[:40]}...' appears before "
            f"'{self.after[:40]}...'."
        )


def load_tests(jsonl_file: str) -> List[BasePDFTest]:
    """
    Load tests from a JSONL file.
    
    Args:
        jsonl_file: Path to the JSONL file containing test definitions.
        
    Returns:
        A list of test objects.
    """
    tests: List[BasePDFTest] = []
    with open(jsonl_file, "r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                test_type = data.get("type")
                if test_type in {TestType.PRESENT.value, TestType.ABSENT.value}:
                    test = TextPresenceTest(**data)
                elif test_type == TestType.ORDER.value:
                    test = TextOrderTest(**data)
                else:
                    raise ValidationError(f"Unknown test type: {test_type}")

                tests.append(test)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e}")
            except (ValidationError, KeyError) as e:
                print(f"Error on line {line_number}: {e}")
            except Exception as e:
                print(f"Unexpected error on line {line_number}: {e}")

    return tests


def save_tests(tests: List[BasePDFTest], jsonl_file: str) -> None:
    """
    Save tests to a JSONL file using asdict for conversion.
    
    Args:
        tests: A list of test objects.
        jsonl_file: Path to the output JSONL file.
    """
    with open(jsonl_file, "w") as file:
        for test in tests:
            file.write(json.dumps(asdict(test)) + "\n")
