import unittest

from toolboxv2 import Singleton


class TestSingleton(unittest.TestCase):

    def test_singleton_instance(self):
        # Create two instances of a class using Singleton metaclass
        class MyClass(metaclass=Singleton):
            pass

        obj1 = MyClass()
        obj2 = MyClass()

        # Check if both objects refer to the same instance
        self.assertIs(obj1, obj2)

    def test_singleton_args_kwargs(self):
        # Create an instance of a class with arguments and keyword arguments
        class MyClass(metaclass=Singleton):
            def __init__(self, arg1, arg2=None, **kwargs):
                self.arg1 = arg1
                self.arg2 = arg2
                self.kwargs = kwargs

        obj1 = MyClass(1, arg2=2, kwarg1='value1')
        obj2 = MyClass(3, arg2=4, kwarg2='value2')

        # Check if arguments and keyword arguments are stored correctly
        self.assertEqual(obj1.arg1, 1)
        self.assertEqual(obj1.arg2, 2)
        self.assertEqual(obj1.kwargs, {'kwarg1': 'value1'})

        self.assertEqual(obj2.arg1, 1)
        self.assertEqual(obj2.arg2, 2)
        self.assertEqual(obj2.kwargs, {'kwarg1': 'value1'})
