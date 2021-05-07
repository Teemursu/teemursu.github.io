---
layout: default
---

## The course in a nutshell

The command-line course teaches students to operate in a Unix enviornment. It is useful especially for language students, as the course also contains information on NLP (Natural Language Processing) and teaches useful ways to process corpus data and write scripts to process large text files (e.g. books). Another important aim of the course is to create and host GitHub Pages, which in the future will work as a kind of portfolio and CV for anyone who wants to pursue a career in coding. 


For the purpose of going through the content of the course and what I learned from it, I will introduce the weekly themes and go through them in order

## Introduction to Command-Line Environments

During the first week we learned some basics on the Unix environment commands and file types. For instance, I learned about binary and non-binary files, commands like cp, touch and cat, as well as about the - (slash) sign which modifies the behavior of Linux shell commands.

Learning about file formats for me was mostly a question about learning the differences between Linux and Windows files, as I was already somewhat familar with Windows file formats.

If one would ever need more information on a command, the man command will provide more information about the other command.

`man mv`

Above, the command will tell more information about the mv command.

A common way to create files is the touch command. The command below creates a file called text.txt into your current working directory.

`touch text.txt`

As said before, the slash sign modifies commands. 

`ls -la`

The command above, ls, normally lists files in the current working directory. By adding -l, the command displays a long list which provides the following information:

* Filetypes 
* File permissions 
* Number of links 
* Owner name 
* Group name
* File size 
* Time of last modification 
* Name of the file or directory. 

The -a lists the hidden files as well. Normally, I have this aliased so that I don't have to write `ls -la` each time, but more on aliases later. 

## Navigating a UNIX System

This week I learned about file and user permissions and the root user, symbolic links, and processes. One of the things that I should probably go back to is the material on remote servers. While I had some familiarity with Taito before, I feel like I forgot some of the deeper understanding on how to work on these remote servers.

The chmod command is used to change permissions of files and directories. It is possible to format these permissions numerically. The chmod numerical is usually a three or four digit number, where the leftmost number concerns the permissions of a user, the number next to it concerns the group and, respectively, the third number concerns the others. The numbers representing permissions vary from 0 to 7, where 7 represents all permissions and 0 none. In fact, these digits are converted to a binary format, which then tells us which permissions are allowed and which are not.

Let us take as an example the following command:

`chmod 360 file1`

In this example, the number 3 represents the permissions of the user. As we convert 3 to binary format, we get 011. In this binary digit, the 0 represents the read permission which is turned off. The 1 next to the 0 is the write permission, which is turned on. The rightmost digit, 1, is also turned on and represents the execute permissions. Hence, this command tells us that the user is allowed to write and execute the file, but not read.

Respectively, the 6 in binary format is 110 and 0 is simply 000. This means that the group is allowed to read and write but not execute, while the others simply have no permissions. 

## Corpus processing

For this week I studied how to process and analyze text files by using commands such as grep -E. Regular expressions were probably my favorite concept to learn in the Introduction to Language Technology course, and I was glad to see them again. 

Still, while I enjoy RegEx immensely, I had troubles with both calculations and further integrating them into a shell environment. In hindsight, I'm not sure why exactly I had difficulties, since the quizes didn't require too much besides RegEx. 

An example of data processing can be seen here, where I calculate the probabiltity of seeing the word 'honey' in a sentence in the book Life of Bee. Here we find all sentences which contain the word "honey." By using \b around the word, we exclude all inflected forms of the word:

`cat life_of_bee.sent | grep -E "\bhoney\b"  | wc -l` 

Then, we divide the result with the total number of sentences, which is an amount we get with:

`wc -l life_of_bee.sent`

To note, the .sent file is the text of the book converted to a sentence per line format. 


## Scripting and UNIX Configuration Files

Next, I learned about scripting. For example, I got to customize my own .bashrc file.

This was the most difficult week for me, as it felt like I sohuld have had at least some practice or familiarity with scripting before. 

Nonetheless, here are some of my own customizations on .bashrc:

Here is a sample of a code that lets us customize the background color of shell:

`export PS1="\u@\[\e[31;46m\]\h\[\e[m\] "`

To change the default BASH prompt from "username@hostname" to my username, and to add username with hostname, I wrote the following code: 

`PS1="temppa> "
export PS1="\u\h "`

In .bashrc, you can define aliases as types of custom shortcuts to represent commands. Here I have some examples of the ones I used in the fourth week.

To clear the terminal:

`alias c='clear'`


Navigating to the course directory was probably my most used command, so I made a shortcut for myself. By typing 'letscode', I was taken to the course directory:

`alias letscode='cd /home/temep/cmdline_course'``


## Installing and Running Programs

For the second to last week I learned about commands such as make. Additionally, while I already had some familarity with Python from another course, I felt like I got a deeper understanding about what exactly pip is.

One of the Python modules I installed this week was the BLLIP parser. After importing the module in python3, we can use the following command to parse the sentence "NLP is fun!":

`rrp.simple_parse("NLP is fun!")`

This gives us the following parse tree:

`'(S1 (S (NP (NNP NLP)) (VP (VBZ is) (NP (NN fun))) (. !)))'`

Here is a more visually appealing presentation of the syntax tree.

![Syntax Tree](/assets/img/parsetree.png)

## Version Control

During this last week, I got to prepare for the final assignment of the course (which basically is this page). I have visited github before, and occasionally downloaded files from github, but I never really knew what it was, or how version control works. 

Normally, when one would save a file, all the previous changes and versions of the file would be lost. Yet, version control is a system that tracks these changes. In this case, GitHub is a web-based service that applies this concept to code: we are now able to manage code (collectively) and see exactly the changes made to the source code that people make.

Some of the most used commands in managing a project on Github are the git add, git commit and git push.

git add adds files to staging area. With --all we can add all the tracked files to the staging area.

`git add --all`

"git commit" commits the changes made to a Git repository. The -m stands for "message" and the message written should describe what kind of changes you have commited to the Git repository.

`git commit -m "Changes made"`

Lastly, the git push moves the code from your local repository to Github, and git pull grabs code from a remote repository to your local repository.

`git push`

`git pull https://github.com/Teemursu/teemursu.github.io`

## Building Webpages using Github Pages

Finally, I have installed and learned to use Jekyll to create this GitHub page. Building a webpage using this method is mostly writing and editing Markdown files (.md). 

By using the Markdown syntax and editing .md files, we are able to format text by **bolding**, *italicizing* and add images or tables. Below is a cheatsheet table of Markdown.

Markdown | Appears as
--- | --- 
`**bold**` | **bold**
`_italics_` | _italics_
``code``| `code`
`![image](linktoimage)` | ![image](https://s3-eu-west-1.amazonaws.com/mordhau-media/spirit/images/1759/1b3874ae216e3f8d1afa08cb4b9e8309.png)
`[link](https://www.google.com)` | [link](https://www.google.com) 
