git config --global push.default simple
git config --global user.name "Github Workflow"
git config --global user.email "moritz@vlooz.de"
set +x
git clone "https://__token__:${GITHUB_TOKEN}@github.com/esa/pyoptgra.git" pyoptgra_gh_pages -q
set -x
cd pyoptgra_gh_pages
git checkout -b gh-pages --track origin/gh-pages;
git rm -fr *;
mv ../doc/sphinx/_build/html/* .;
touch .nojekyll
git add .nojekyll
git add *
# We assume here that a failure in commit means that there's nothing
# to commit.
git commit -m "Update Sphinx documentation commit ${GITHUB_SHA} [skip ci]." || exit 0
PUSH_COUNTER=0
until git push -q
do
    git pull -q
    PUSH_COUNTER=$((PUSH_COUNTER + 1))
    if [ "$PUSH_COUNTER" -gt 3 ]; then
        echo "Push failed, aborting.";
        exit 1;
    fi
done

set +e
set +x