#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

def observe_steps(sub_or_sol):
  # --0-> [0] <-0-> [1] <-0-> [2] --1-> [3] <-0--
  mdp_data = sub_or_sol.initialize_mdp_data(num_states = 4)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 0, 0, 0, 0)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 0, 1, 1, 0)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 1, 0, 0, 0)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 1, 1, 2, 0)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 2, 0, 1, 1)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 2, 1, 3, 1)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 3, 0, 3, 0)
  sub_or_sol.update_mdp_transition_counts_sum_reward(mdp_data, 3, 1, 3, 0)
  return mdp_data

#########
# TESTS #
#########


class Test_1(GradedTestCase):

  def check_valid_mdp_data(self, mdp_data, n_states):
    # MDP Data must be a dict with the following keys and value types/shapes:
    # {
    #     'transition_probs': np.ndarray, dtype=np.float64, shape=(num_states, num_actions, num_states).
    #     'transition_counts': np.ndarray, dtype=np.float64, shape=(num_states, num_actions, num_states).
    #
    #     'avg_reward': np.ndarray, dtype=np.float64, shape=(num_states,).
    #     'sum_reward': np.ndarray, dtype=np.float64, shape=(num_states,).
    #
    #     'value': np.ndarray, dtype=np.float64, shape=(num_states,).
    #     'num_states': Int
    #     }
    n_actions = 2
    self.assertIsInstance(mdp_data, dict)

    self.assertIn('transition_probs', mdp_data)
    self.assertIsInstance(mdp_data['transition_probs'], np.ndarray)
    self.assertEqual(mdp_data['transition_probs'].dtype, np.float64)
    self.assertEqual(mdp_data['transition_probs'].shape, (n_states, n_actions, n_states))

    self.assertIn('transition_counts', mdp_data)
    self.assertIsInstance(mdp_data['transition_counts'], np.ndarray)
    self.assertEqual(mdp_data['transition_counts'].dtype, np.float64)
    self.assertEqual(mdp_data['transition_counts'].shape, (n_states, n_actions, n_states))

    self.assertIn('avg_reward', mdp_data)
    self.assertIsInstance(mdp_data['avg_reward'], np.ndarray)
    self.assertEqual(mdp_data['avg_reward'].dtype, np.float64)
    self.assertEqual(mdp_data['avg_reward'].shape, (n_states, ))

    self.assertIn('sum_reward', mdp_data)
    self.assertIsInstance(mdp_data['sum_reward'], np.ndarray)
    self.assertEqual(mdp_data['sum_reward'].dtype, np.float64)
    self.assertEqual(mdp_data['sum_reward'].shape, (n_states, ))

    self.assertIn('value', mdp_data)
    self.assertIsInstance(mdp_data['value'], np.ndarray)
    self.assertEqual(mdp_data['value'].dtype, np.float64)
    self.assertEqual(mdp_data['value'].shape, (n_states, ))

    self.assertIn('num_states', mdp_data)
    self.assertIsInstance(mdp_data['num_states'], int)
    
  @graded()
  def test_00(self):
    """1-0-basic: Evaluating initialize_mdp_data() returns correct data types and shapes."""
    n_states = 10
    self.check_valid_mdp_data(submission.initialize_mdp_data(n_states), n_states)

  @graded()
  def test_01(self):
    """1-1-basic:  Evaluating function initialize_mdp_data() (simple)."""
    n_states = 1
    mdp_data = submission.initialize_mdp_data(n_states)
    self.check_valid_mdp_data(mdp_data, n_states)
    self.assertTrue((mdp_data['transition_probs'] == np.array([[[1.], [1.]]], dtype=np.float64)).all())
    self.assertTrue((mdp_data['transition_counts'] == np.array([[[0.], [0.]]], dtype=np.float64)).all())
    self.assertTrue((mdp_data['avg_reward'] == np.array([0.], dtype=np.float64)).all())
    self.assertTrue((mdp_data['sum_reward'] == np.array([0.], dtype=np.float64)).all())
    self.assertEqual(mdp_data['num_states'], 1)

  @graded(is_hidden=True, after_published=False)
  def test_02i(self):
    """1-2i-hidden:  Evaluating function initialize_mdp_data() (transition_probs, complex)."""
    n_states = 163
    mdp_data = submission.initialize_mdp_data(n_states)
    self.check_valid_mdp_data(mdp_data, n_states)
    self.assertTrue((self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.initialize_mdp_data(163)['transition_probs']) ==
                     mdp_data['transition_probs']).all())

  @graded(is_hidden=True, after_published=False)
  def test_02ii(self):
    """1-2ii-hidden:  Evaluating function initialize_mdp_data() (transition_counts, complex)."""
    n_states = 163
    mdp_data = submission.initialize_mdp_data(n_states)
    self.check_valid_mdp_data(mdp_data, n_states)
    self.assertTrue((self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.initialize_mdp_data(163)['transition_counts']) ==
                     mdp_data['transition_counts']).all())

  @graded(is_hidden=True, after_published=False)
  def test_02iii(self):
    """1-2iii-hidden:  Evaluating function initialize_mdp_data() (avg_reward, complex)."""
    n_states = 163
    mdp_data = submission.initialize_mdp_data(n_states)
    self.check_valid_mdp_data(mdp_data, n_states)
    self.assertTrue((self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.initialize_mdp_data(163)['avg_reward']) ==
                     mdp_data['avg_reward']).all())

  @graded(is_hidden=True, after_published=False)
  def test_02iv(self):
    """1-2iv-hidden:  Evaluating function initialize_mdp_data() (sum_reward, complex)."""
    n_states = 163
    mdp_data = submission.initialize_mdp_data(n_states)
    self.check_valid_mdp_data(mdp_data, n_states)
    self.assertTrue((self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.initialize_mdp_data(163)['sum_reward']) ==
                     mdp_data['sum_reward']).all())

  @graded(is_hidden=True, after_published=False)
  def test_02v(self):
    """1-2v-hidden:  Evaluating function initialize_mdp_data() (num_states, complex)."""
    n_states = 163
    mdp_data = submission.initialize_mdp_data(n_states)
    self.check_valid_mdp_data(mdp_data, n_states)
    self.assertEqual(self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.initialize_mdp_data(163)['num_states']),
                     mdp_data['num_states'])

  @graded()
  def test_03(self):
    """1-3-basic:  Simple check of choose_action."""
    n_states = 2
    mdp_data = submission.initialize_mdp_data(n_states)
    self.check_valid_mdp_data(mdp_data, n_states)
    mdp_data['transition_probs'] = np.array([[[1., 0.],
                                              [0., 1.]],
                                             [[1., 0.],
                                              [0., 1.]]])
    mdp_data['value'] = np.array([0,1],dtype=np.float64)
    action = submission.choose_action(state=0, mdp_data=mdp_data)
    self.check_valid_mdp_data(mdp_data, n_states)
    self.assertEqual(action, 1)

  @graded(is_hidden=True, after_published=False)
  def test_04(self):
    """1-4-hidden:  Checking choose_action() with multiple states and complete value function."""
    solution_mdp_data = self.run_with_solution_if_possible(submission, observe_steps)
    self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.update_mdp_transition_probs_avg_reward(solution_mdp_data))
    self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.update_mdp_value(solution_mdp_data, 0.01, 0.99))

    action_0 = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.choose_action(state=0, mdp_data=solution_mdp_data))
    action_1 = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.choose_action(state=1, mdp_data=solution_mdp_data))
    action_2 = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.choose_action(state=2, mdp_data=solution_mdp_data))
    action_3 = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.choose_action(state=3, mdp_data=solution_mdp_data))

    self.assertEqual(submission.choose_action(state=0, mdp_data = solution_mdp_data), action_0)
    self.assertEqual(submission.choose_action(state=1, mdp_data = solution_mdp_data), action_1)
    self.assertEqual(submission.choose_action(state=2, mdp_data = solution_mdp_data), action_2)

  @graded()
  def test_05(self):
    """1-5-basic:  Checking no steps in update_mdp_transition_probs_avg_reward() (will pass if not implemented)."""
    mdp_data = {}
    mdp_data['transition_probs'] = np.array([[[1.], [1.]]],dtype=np.float64)
    mdp_data['transition_counts'] = np.array([[[0.], [0.]]],dtype=np.float64)
    mdp_data['avg_reward'] = np.array([0.],dtype=np.float64)
    mdp_data['sum_reward'] = np.array([0.],dtype=np.float64)
    mdp_data['num_states'] = 1
    submission.update_mdp_transition_probs_avg_reward(mdp_data)
    self.assertTrue((mdp_data['transition_probs'] == np.array([[[1.], [1.]]],dtype=np.float64)).all())
    self.assertTrue((mdp_data['avg_reward'] == np.array([0.],dtype=np.float64)).all())

  @graded()
  def test_06(self):
    """1-6-basic:  Checking calculations in update_mdp_transition_probs_avg_reward() (1 state, 1 step)."""
    mdp_data = {}
    mdp_data['transition_probs'] = np.array([[[1.], [1.]]],dtype=np.float64)
    mdp_data['transition_counts'] = np.array([[[1.], [0.]]],dtype=np.float64)
    mdp_data['avg_reward'] = np.array([0.],dtype=np.float64)
    mdp_data['sum_reward'] = np.array([1.],dtype=np.float64)
    mdp_data['num_states'] = 1
    submission.update_mdp_transition_probs_avg_reward(mdp_data)
    self.assertTrue((mdp_data['transition_probs'] == np.array([[[1.], [1.]]],dtype=np.float64)).all())
    self.assertTrue((mdp_data['avg_reward'] == np.array([1.],dtype=np.float64)).all())

  @graded()
  def test_07(self):
    """1-7-basic:  Checking calculations in update_mdp_transition_probs_avg_reward() (1 state, 1 step, 1 reward)."""
    mdp_data = {}
    mdp_data['transition_probs'] = np.array([[[1.], [1.]]],dtype=np.float64)
    mdp_data['transition_counts'] = np.array([[[1.], [0.]]],dtype=np.float64)
    mdp_data['avg_reward'] = np.array([0.],dtype=np.float64)
    mdp_data['sum_reward'] = np.array([1.],dtype=np.float64)
    mdp_data['num_states'] = 1
    submission.update_mdp_transition_probs_avg_reward(mdp_data)
    self.assertTrue((mdp_data['transition_probs'] == np.array([[[1.], [1.]]],dtype=np.float64)).all())
    self.assertTrue((mdp_data['avg_reward'] == np.array([1.],dtype=np.float64)).all())

  @graded(is_hidden=True, after_published=False)
  def test_08i(self):
    """1-8i-hidden:  Checking transition_probs calculations in update_mdp_transition_probs_avg_reward() (multiple steps and rewards)."""
    mdp_data = self.run_with_solution_if_possible(submission, observe_steps)
    submission.update_mdp_transition_probs_avg_reward(mdp_data)

    solution_mdp_data = self.run_with_solution_if_possible(submission, observe_steps)
    self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.update_mdp_transition_probs_avg_reward(solution_mdp_data))

    self.assertTrue((mdp_data['transition_probs'] == solution_mdp_data['transition_probs']).all())

  @graded(is_hidden=True, after_published=False)
  def test_08ii(self):
    """1-8ii-hidden:  Checking avg_reward calculations in update_mdp_transition_probs_avg_reward() (multiple steps and rewards)."""

    mdp_data = self.run_with_solution_if_possible(submission, observe_steps)
    submission.update_mdp_transition_probs_avg_reward(mdp_data)

    solution_mdp_data = self.run_with_solution_if_possible(submission, observe_steps)
    self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.update_mdp_transition_probs_avg_reward(solution_mdp_data))

    self.assertTrue((mdp_data['avg_reward'] == solution_mdp_data['avg_reward']).all())

  @graded()
  def test_09(self):
    """1-9-basic:  Checking value iteration algorithm returns the correct types and shapes"""
    mdp_data = observe_steps(submission)
    submission.update_mdp_transition_probs_avg_reward(mdp_data)
    submission.update_mdp_value(mdp_data, 0.01, 0.99)
    self.check_valid_mdp_data(mdp_data, 4)

  @graded()
  def test_10(self):
    """1-10-basic: Checking simple, two-step value iteration."""
    mdp_data = {}
    mdp_data['transition_probs'] = np.array([[[0., 1., 0.],
                                              [0., 1., 0.]],
                                             [[0., 0., 1.],
                                              [0., 0., 1.]],
                                             [[0., 0., 1.],
                                              [0., 0., 1.]]],dtype=np.float64)
    mdp_data['transition_counts'] = np.array([[[0., 1., 0.],
                                               [0., 1., 0.]],
                                              [[0., 0., 1.],
                                               [0., 0., 1.]],
                                              [[0., 0., 1.],
                                               [0., 0., 1.]]],dtype=np.float64)
    mdp_data['avg_reward'] = np.array([0.,1.,0.],dtype=np.float64)
    mdp_data['sum_reward'] = np.array([0.,2.,0.],dtype=np.float64)
    mdp_data['num_states'] = 3
    mdp_data['value'] = np.array([0.,0.,0.], dtype=np.float64)
    submission.update_mdp_value(mdp_data, 0.01, 0.99)
    self.assertTrue((mdp_data['value'] == np.array([.99,1.,0.], dtype=np.float64)).all())

  @graded(is_hidden=True, after_published=False)
  def test_11(self):
    """1-11-hidden:  Checking value iteration with complex environment."""
    sub_or_sol = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol)
    mdp_data = observe_steps(sub_or_sol)
    v = mdp_data['value']
    sub_or_sol.update_mdp_transition_probs_avg_reward(mdp_data)
    sub_or_sol.update_mdp_value(mdp_data, 0.01, 0.99)
    solution_value = mdp_data['value']

    mdp_data = observe_steps(sub_or_sol)
    mdp_data['value'] = v
    sub_or_sol.update_mdp_transition_probs_avg_reward(mdp_data)
    submission.update_mdp_value(mdp_data, 0.01, 0.99)
    submission_value = mdp_data['value']

    self.assertTrue((solution_value == submission_value).all())

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
