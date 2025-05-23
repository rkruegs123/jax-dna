### Getting Started

To start working, first [fork the
repository](https://github.com/rkruegs123/jax-dna/fork), clone it to your local
machine, and create a new branch for your changes. You can do this with the
following commands:

```bash
git clone https://github.com/<username-or-org>/jax-dna.git
cd jax-dna
git checkout -b <branch-name>
```

Now that you are in your own branch in your fork you can start making changes!

### Getting Ready to Submit

If you are adding a new feature or changing functionality, please make sure that
you have added tests for your changes. We use
[pytest](https://docs.pytest.org/en/stable/) to run our tests. You can find
examples of tests in a subdirectory called `tests`, wherever you are adding
functionality. JAX-DNA currently requires at least 90% test coverage in the
overall project code and 80% test coverage in the code that has been added.

JAX-DNA uses `tox` to manage development and testing.

`tox` is used to:
- Run tests (`tox -e test`)
- Run the formatter (`tox -e format`)
- Run the linter (`tox -e check-style`)
- Run the security checks (`tox -e check-security`)
- Build the package (`tox -e build`)
- Build the documentation (`tox -e build-docs`)
- Run all checks (`tox`)

After you have run the tests, you can check the coverage report with the
following command, starting from the root of the repository:

```bash
cd .coverage.html
python -m http.server -p 8080
```

Then open your browser and navigate to the displayed URL, which could be
something like `http://localhost:8080/` or `http://0.0.0.0:8080`. You should
see a list of the files and their current coverage.

### Pushing Your Changes

The first time you push your commits you will need to set the
origin for your branch:

```bash
git push -u origin <branch_name>
```

### Merging your changes back to the main repository

Once you are happy with your changes and have pushed them to your fork, you can
create a pull request to merge your changes back into the main repository. There
should be a link in the output from the `push` command that you can click that
looks like:

```
https://github.com/<username-or-org>/jax-dna/pull/new/<branch_name>
```

If you don't see that link, you can go to the main repository and click on the
"Pull Requests" tab.

