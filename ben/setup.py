import os, platform, uuid
from setuptools import setup, find_packages

if platform.platform().startswith('Linux'):
    req_lines = [line.strip() for line in open('requirements.txt').readlines()]
    install_reqs = list(filter(None, req_lines))
else:
    from pip.req import parse_requirements
    install_requires = parse_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"), None, None,
                                          None, uuid.uuid1())
    install_reqs = [str(r.req) for r in install_requires]

setup_kwargs = {
    'name': "lungai",
    'version': "0.1.0",
    'packages': find_packages("src"),
    # 'scripts':  ['py/pydio.py'],
    'package_dir': {'': 'src'},
    'install_requires': install_reqs,

    "package_data": {
        'sdk': [  'ui/res/templates/*.html',
                  'ui/res/images/*.png',
                  'ui/res/static/css/*.css',
                  'ui/res/static/js/*.js',
               ]
    },

    # metadata for upload to PyPI
    'author': "Ben Neely",
    'author_email': "nigelneely@gmail.com",
    'description': "Tools used to train computers to ingest, segment, and annotate lungmap images.",
    'license': "BSD",
    'keywords': "lungmap",
    'url': "https://github.com/duke-lungmap-team/",

    # Create an entry programs for this package
    "entry_points": {
        'console_scripts': [
            'lungmapai = sdk.main:main',
        ],
    },
}

setup(**setup_kwargs)