from setuptools import setup

setup(name="epsnoise",
      description="Simulate pixel noise for weak-lensing ellipticity measurements",
      long_description="Simulate pixel noise for weak-lensing ellipticity measurements",
      version="0.1",
      license="MIT",
      author="Peter Melchior",
      author_email="peter.m.melchior@gmail.com",
      py_modules=["epsnoise"],
      url="https://github.com/pmelchior/epsnoise",
      requires=["numpy", "scipy"]
)

