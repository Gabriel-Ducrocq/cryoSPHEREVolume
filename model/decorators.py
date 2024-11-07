import functools
from time import time

def timing(func):
	"""Timing the execution of a function"""
	@functools.wraps(func)
	def wrapper_function(*args, **kwargs):
		start = time()
		output = func(*args, **kwargs)
		end = time()
		print(f"Execution of function {func.__name__} : {end-start} seconds")
		return output

	return wrapper_function
