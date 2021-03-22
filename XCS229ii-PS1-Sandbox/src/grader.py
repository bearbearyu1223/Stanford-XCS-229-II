#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########

class Test_1a(GradedTestCase):
  @graded()
  def test_0(self):
    """1a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1a()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1a', lambda f: set([choice.lower() for choice in f()]))
class Test_1b(GradedTestCase):
  @graded()
  def test_0(self):
    """1b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1b()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1b', lambda f: set([choice.lower() for choice in f()]))
class Test_1c(GradedTestCase):
  @graded()
  def test_0(self):
    """1c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1c()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1c-1-hidden:  Multiple choice response."""
    self.assertTrue(True)
    # self.compare_with_solution_or_wait(submission, 'multiple_choice_1c', lambda f: set([choice.lower() for choice in f()]))
class Test_1d(GradedTestCase):
  @graded()
  def test_0(self):
    """1d-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1d()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1d-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1d', lambda f: set([choice.lower() for choice in f()]))
class Test_1e(GradedTestCase):
  @graded()
  def test_0(self):
    """1e-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1e()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1e-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1e', lambda f: set([choice.lower() for choice in f()]))
class Test_1f(GradedTestCase):
  @graded()
  def test_0(self):
    """1f-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1f()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1f-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1f', lambda f: set([choice.lower() for choice in f()]))
class Test_1g(GradedTestCase):
  @graded()
  def test_0(self):
    """1g-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1g()])
    self.assertTrue(response.issubset(set(['a','b','c','d','e'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),2, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1g-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1g', lambda f: set([choice.lower() for choice in f()]))
class Test_1h(GradedTestCase):
  @graded()
  def test_0(self):
    """1h-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1h()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),2, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1h-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1h', lambda f: set([choice.lower() for choice in f()]))
class Test_1i(GradedTestCase):
  @graded()
  def test_0(self):
    """1i-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1i()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1i-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1i', lambda f: set([choice.lower() for choice in f()]))
class Test_1j(GradedTestCase):
  @graded()
  def test_0(self):
    """1j-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1j()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1j-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1j', lambda f: set([choice.lower() for choice in f()]))

class Test_2a(GradedTestCase):
  @graded()
  def test_0(self):
    """2a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2a()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2a', lambda f: set([choice.lower() for choice in f()]))
class Test_2b(GradedTestCase):
  @graded()
  def test_0(self):
    """2b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2b()])
    self.assertTrue(response.issubset(set(['a','b'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2b', lambda f: set([choice.lower() for choice in f()]))
class Test_2c(GradedTestCase):
  @graded()
  def test_0(self):
    """2c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2c()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2c', lambda f: set([choice.lower() for choice in f()]))
class Test_2d(GradedTestCase):
  @graded()
  def test_0(self):
    """2d-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2d()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2d-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2d', lambda f: set([choice.lower() for choice in f()]))
class Test_2e(GradedTestCase):
  @graded()
  def test_0(self):
    """2e-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2e_i()])
    self.assertTrue(response.issubset(set(['a','b'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

    response = set([choice.lower() for choice in submission.multiple_choice_2e_ii()])
    self.assertTrue(response.issubset(set(['a','b'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

    response = set([choice.lower() for choice in submission.multiple_choice_2e_iii()])
    self.assertTrue(response.issubset(set(['a','b'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

    response = set([choice.lower() for choice in submission.multiple_choice_2e_iv()])
    self.assertTrue(response.issubset(set(['a','b'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_i(self):
    """2e-i-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2e_i', lambda f: set([choice.lower() for choice in f()]))

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_ii(self):
    """2e-ii-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2e_ii', lambda f: set([choice.lower() for choice in f()]))

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_iii(self):
    """2e-iii-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2e_iii', lambda f: set([choice.lower() for choice in f()]))

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_iv(self):
    """2e-iv-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2e_iv', lambda f: set([choice.lower() for choice in f()]))

class Test_3a(GradedTestCase):
  @graded()
  def test_0(self):
    """3a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_3a()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """3a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_3a', lambda f: set([choice.lower() for choice in f()]))
class Test_3b(GradedTestCase):
  @graded()
  def test_0(self):
    """3b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_3b()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """3b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_3b', lambda f: set([choice.lower() for choice in f()]))
class Test_3c(GradedTestCase):
  @graded()
  def test_0(self):
    """3c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_3c()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """3c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_3c', lambda f: set([choice.lower() for choice in f()]))

class Test_4a(GradedTestCase):
  @graded()
  def test_0(self):
    """4a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4a()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4a', lambda f: set([choice.lower() for choice in f()]))
class Test_4b(GradedTestCase):
  @graded()
  def test_0(self):
    """4b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4b()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4b', lambda f: set([choice.lower() for choice in f()]))
class Test_4c(GradedTestCase):
  @graded()
  def test_0(self):
    """4c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4c()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4c', lambda f: set([choice.lower() for choice in f()]))

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)
