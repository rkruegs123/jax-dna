## How to Contribute

First [fork](https://github.com/rkruegs123/jax-dna-dev/fork) the repository.

Then clone the forked version of the repository onto
your local machine:

```bash
git clone https:www.github.com/<your_username>/jax-dna-dev.git
```

After cloning, cd into the repository and make a new branch:

```bash
cd ./jax-dna-dev
git checkout -b <branch_name>
```

You are now working in the new branch you created. You can
verify this by running `git status`.

Now you can fix bugs, add new features, etc.

The first time you push your commits you will need to set the
origin for your branch:

```bash
git push -u origin <branch_name>
```

When you are ready contribute your changes back to the
repository, you need to do the following:

1. Install tox, `pip install tox`
2. Format your code to conform with the `jax_dna` style
   guide lines: `tox -e format`
3. Run the linter to check for any other style, formatting
   issues, or potential bugs `tox -e check-style`
4. Finally, run the tests and check the test coverage:
   `tox -e test`

**NOTE:** Any new code contributions need to have at least
80% code coverage in order to pass the Github CI tests and
be approved for merge.

If all the checks pass, congrats! You're ready to submit a
pull request to the main `jax_dna` repo.

You can do this from either your repo page or from the
jax-dna repo page.
