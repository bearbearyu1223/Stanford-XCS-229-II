#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, time
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy

# Import student submission
import submission
numpy.random.seed(0)

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########

# By convention, test classes are used for each subquestion (1a, 1b, 2a, 2b, etc.)
# Tests within a class are numbered, starting at 0: test_0(), test_1(), etc.
# Tests must have a docstring.  The docstring must be in the following format:
# """test_name:  test_description"""
# The test name should be: subquestion-test_num-{basic/hidden}
#
# Access the solutions with the following method:
#   result = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.func(test_args))
#
# In the supplied function, the first and only argument is either the submission
# or the solution, depending on whether the autograder is run locally or with
# the solutions present.


class Test_1b(GradedTestCase):
  def basic_test(self, submission_op, expected_op, func_being_tested):
    """
    Basic shape and data type test
    Args:
        submission_op: student submission output of the method being tested
        expected_op: expected output
        func_being_tested: name of the function being tested

    Returns: None

    """
    # use assertIsInstance instead
    self.assertTrue(isinstance(submission_op, type(expected_op)),
                    msg="Expected output of {func_being_tested} function to be of type : "
                        "{expected_op_type} but got {submission_op_type}"
                    .format(func_being_tested=func_being_tested,
                            expected_op_type=type(expected_op),
                            submission_op_type=type(submission_op)))

    self.assertTrue(expected_op.shape == submission_op.shape,
                    msg="Expected output shape of {func_being_tested} function to be : "
                        "{expected_op_shape} but got {submission_op_shape}"
                    .format(func_being_tested=func_being_tested,
                            expected_op_shape=expected_op.shape,
                            submission_op_shape=submission_op.shape))

  @graded(timeout=30)
  def test_00(self):
    """1b-0-basic:  Evaluating softmax() output"""
    x = numpy.array([[0.40740357, 0.46248366],
                     [0.73424676, 0.52653602],
                     [0.21102139, 0.34013982],
                     [0.98344458, 0.73088265]])

    submission_op = submission.softmax(x)
    expected_op = numpy.array([[0.48623346, 0.51376654],
                               [0.55174179, 0.44825821],
                               [0.46776516, 0.53223484],
                               [0.56280698, 0.43719302]])

    self.basic_test(submission_op=submission_op, expected_op=expected_op, func_being_tested="softmax()")

    self.assertTrue(numpy.allclose(expected_op, submission_op),
                    msg="Expected output of softmax() function to be {} but got {}".format(expected_op,
                                                                                           submission_op))

  @graded(timeout=30)
  def test_01i(self):
    """1b-1i-basic:  Evaluating forward_pass() output datatype and shape"""
    S = numpy.array([[0.0, 1.0, 1.0, 0.0]])
    W = numpy.array([[0.71186985, 0.80105879],
                     [0.40029088, 0.16408907],
                     [0.24686674, 0.37594635],
                     [0.50711907, 0.47308381]])

    submission_op = submission.forward_pass(W, S)
    expected_op = numpy.array([[0.52675497, 0.47324503]])
    self.basic_test(submission_op=submission_op, expected_op=expected_op, func_being_tested="forward_pass()")

  @graded(timeout=30, is_hidden=True, after_published=False)
  def test_01ii(self):
    """1b-1ii-hidden:  Evaluating forward_pass() output"""
    S = numpy.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
    W = numpy.array([[0.54642436, 0.8941891],
                     [0.6116459, 0.91185742],
                     [0.40735244, 0.16564978],
                     [0.614072, 0.10781311],
                     [0.19485619, 0.03428044]])

    submission_op = submission.forward_pass(W, S)
    expected_op = self.run_with_solution_if_possible(submission,
                                                     lambda sub_or_sol: sub_or_sol.forward_pass(W, S))
    self.assertTrue(numpy.allclose(expected_op, submission_op),
                    msg="Output of forward_pass() function does not match expected value\nYour output:\n"
                    .format(submission_op))

  @graded(timeout=30)
  def test_02i(self):
    """1b-2i-basic:  Evaluating policy_gradient() output shape and datatype"""

    W = numpy.array(([[0.48435543, 0.87405626],
                      [0.9785218, 0.97894915],
                      [0.94119834, 0.01011787]]))
    S = numpy.array([[1.0, 0.0, 1.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 1.0, 0.0]])
    A = numpy.array([1, 1, 0])
    G = numpy.array([0.64159167, 0.51844126, 0.8613691])
    learning_rate = 0.01

    submission_op = submission.policy_gradient(W, S, A, G, learning_rate)
    expected_op = numpy.array([[-0.00219649, 0.00219649], [0.00513646, -0.00513646], [-0.00733295, 0.00733295]])
    self.basic_test(submission_op=submission_op, expected_op=expected_op, func_being_tested="policy_gradient()")

  @graded(timeout=30, is_hidden=True, after_published=False)
  def test_02ii(self):
    """1b-2ii-hidden:  Evaluating policy_gradient() output"""

    # TODO - initialize the right value
    W = numpy.array(([[0.86514535, 0.54550479],
                      [0.96762212, 0.46626473],
                      [0.02265237, 0.66559754]]))
    S = numpy.array([[0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0]])
    A = numpy.array([0, 1, 0])
    G = numpy.array([0.58153841, 0.21730104, 0.73679393])
    learning_rate = 0.01

    submission_op = submission.policy_gradient(W, S, A, G, learning_rate)
    expected_op = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.policy_gradient(W, S, A, G, learning_rate))
    self.assertTrue(numpy.allclose(expected_op, submission_op),
                    msg="Output of policy_gradient() does not match expected value\nYour output:\n"
                    .format(submission_op))

  @graded(timeout=30)
  def test_03(self):
    """1b-3-basic:  Evaluating init_policy_data() output"""
    num_state_params = 4
    num_actions = 2

    submission_op = submission.init_policy_data(num_state_params, num_actions)
    # expected_op in this case may not be exactly equal to submission_op. We only evaluate for shape and the data type
    expected_op = {
      'W': numpy.array([[1.76405235,  0.40015721],
                        [0.97873798,  2.2408932],
                        [1.86755799, -0.97727788],
                        [0.95008842, -0.15135721],
                        [-0.10321885,  0.4105985]]),
      'episode': [],
      'history': []}

    self.assertIn('W', submission_op)
    self.assertIn('episode', submission_op)
    self.assertIn('history', submission_op)

    self.basic_test(submission_op=submission_op['W'], expected_op=expected_op['W'],
                    func_being_tested="init_policy_data()")

  @graded(timeout=30)
  def test_04(self):
    """1b-4-basic:  Evaluating choose_action() output"""
    state = numpy.array([1., 1., 0., 0., 0.])
    policy_data = {'W': numpy.array([[0.06381596, -0.12089246,  0.02620287, -0.6438723],
                                     [-0.19561564,  0.84086584,  0.98102769,  0.03692181],
                                     [-1.33627622,  2.09333675,  0.77533065, -0.11594722],
                                     [0.7757198, -0.09771534,  0.25640622,  0.35121989],
                                     [-0.97432407, -1.49746049,  0.04341171,  0.52956114]]),
                   'episode': [],
                   'history': []}
    num_actions = 4

    submission_op = submission.choose_action(state, policy_data, num_actions)
    self.assertTrue(numpy.issubdtype(type(submission_op), numpy.integer))
    self.assertTrue(submission_op in set(numpy.arange(num_actions)))

  @graded(timeout=30)
  def test_05(self):
    """1b-5-basic:  Evaluating record_transition()"""
    policy_data = {'W': numpy.array([[-0.91033536, -1.46960675, -0.78136995],
                                     [0.27055657, -1.17676548, -0.21280714],
                                     [-0.01503552,  0.27920773,  1.81210538],
                                     [-0.5006686 ,  0.46293075, -1.46280086]]),
                   'episode': [],
                   'history': []}
    prior_state = numpy.array([1., 0., 0., 1.])
    action = 2
    reward = 0.0
    posterior_state = numpy.array([1., 1., 0., 0.])

    policy_data_updated = copy.deepcopy(policy_data)
    submission.record_transition(policy_data_updated, prior_state, action, reward, posterior_state)

    self.assertTrue(len(policy_data_updated['episode']) - len(policy_data['episode']) == 1)
    # Note : You may not need all of the data provided to this record_transition method (s, a, r, s')
    self.assertTrue(len(policy_data_updated['episode'][-1]) == 3)

  @graded(timeout=30)
  def test_06i(self):
    """1b-6i-basic:  Evaluating accumulate_discounted_future_rewards() output datatype and shape"""
    R = numpy.array([0., 0., 0., -1., 0., 0., 0.,  0., 0., -1.])
    gamma = 0.5

    submission_op = submission.accumulate_discounted_future_rewards(R, gamma)
    expected_op = numpy.array([-0.12695312, -0.25390625, -0.5078125, -1.015625, -0.03125, -0.0625,
                               -0.125, -0.25, -0.5, -1.])

    self.basic_test(submission_op=submission_op, expected_op=expected_op,
                    func_being_tested='accumulate_discounted_future_rewards()')

  @graded(timeout=30, is_hidden=True, after_published=False)
  def test_06ii(self):
    """1b-6ii-hidden:  Evaluating accumulate_discounted_future_rewards() output"""
    R = numpy.array([0., 0., -1., -1., 0., -1.])
    gamma = 0.3

    submission_op = submission.accumulate_discounted_future_rewards(R, gamma)
    expected_op = self.run_with_solution_if_possible(submission,
                                                     lambda sub_or_sol:
                                                     sub_or_sol.accumulate_discounted_future_rewards(R, gamma))
    self.assertTrue(numpy.allclose(expected_op, submission_op),
                    msg="Output of policy_gradient() function does not match expected value\nYour output:\n{}"
                    .format(submission_op))


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
