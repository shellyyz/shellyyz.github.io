---
layout:     post
title:      Use Git to Edit Files on Local Devices
subtitle:   keep updating commands as be more familiar
date:       2018-11-25
author:     shellyyz
description: some common commands in Git.
# header-img: img/post-bg-ios9-web.jpg -->
catalog: 	 true
tags:
    - Git
    - Commands
---
# Use Git to Edit Files on Local Devices
Here are some common commands in Git.

### Example
At the beginning, a simplest but most common example is showed.
```
git init
git add [file_name]
git commit -m [file_name]
git push origin [current_branch]:[branch_to_push]
```
----
### Edit a file and push it
If there is a pushed file, after editing the changes is modified but not staged for commit, so we need to perform
```
git add [file_name]
```
Then the file's status is modified and to be commited, and we can commit it by
```
git commit -m [file_name]
```
Finally we can push it by
```
git push
```
----
### Create a repository
#### create a new repository
Create a new directory, open it and perform a
```
git init
```
to create a new git reporsitory.

#### copy a local repository
Create a new directory, open it and perform a
```
git clone C:/Farm1/ShellFarm/shellyyz.github.io
```
to copy a local existing git reporsitory.

#### copy a remote repository
Open a directory and perform a
```
git clone http://your_url
```
to copy a reporsitory on the website.
----
### Create/ Switch to/ Merge/ Delete a branch
#### list branches
List all the local branches
```
git branch
```
List all the remote branches
```
git branch -r
```
List all the local and remote branches
```
git branch -a
```
#### create a new branch
Create a new branch but still remain in the current branch
```
git branch [branch_name]
```
Create a new branch and switch to the new branch
```
git checkout -b [branch_name]
```
#### switch to a branch
Switch to the specified branch
```
git checkout [branch_name]
```
Switch to the last branch
```
git chechout -
```
#### merge a branch to the current branch
```
git merge [branch_name]
```
#### delete a branch
First ensure the current branch is not the branch to delete, otherwise switch to another branch. Then perform
```
git branch -d [branch_name]
```
----
### Delete files
If want to delete a pushed file from cache, which means don't want to track it anymore, perform
```
git rm --cache [file_name]
git commit -m "delete file"
git push
```
If want to delete a pushed file from repository as well as local, perform
```
git rm --f [file_name]
git commit -m "delete file"
git push
```
