# Introduction

A template for projects that build off of iterativenn simple.
# Notes

## Submodule
This includes iterativenn as a submodule.  To clone the submodule use the following command:

```bash
git submodule update --init --recursive
``` 

[[https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-to-clone-a-git-repository-with-submodules-init-and-update]]

Git clone with submodules

The list of steps required to clone a Git repository with submodules is:

    Issue a git clone command on the parent repository.
    Issue a git submodule init command.
    Issue a git submodule update command.

Git init and update alternative

There is actually an alternative to going through these three steps. You can use the –recurse-submodules switch on the clone. This approach, shown below, might be easier.

git clone --recurse-submodules https://github.com/..