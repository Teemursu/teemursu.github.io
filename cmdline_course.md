---
layout: default
---

## The course in a nutshell

The command-line course teaches students to operate in a Unix enviornment. 

For the purpose of going through the content of the course and what I learned from it, I will introduce the weekly themes and go through them in order

## Week 1: Introduction to Command-Line Environments

During the first week we learned some basics on the Unix environment commands and file types. For instance, I learned about binary and non-binary files, commands like cp, touch and cat, as well as about the - (slash) sign which modifies the behavior of Linux shell commands.

Learning about file formats for me was mostly a question about learning the differences between Linux and Windows files, as I was already somewhat familar with Windows file formats.

## Week 2: Navigating a UNIX System

This week I learned about file and user permissions and the root user, symbolic links, and processes. One of the things that I should probably go back to is the material on remote servers. While I had some familiarity with Taito before, I feel like I forgot some of the deeper understanding on how to work on these remote servers.

## Week 3: Corpus processing

For this week I studied how to process and analyze text files by using commands such as grep -E. Regular expressions were probably my favorite concept to learn in the Introduction to Language Technology course, and I was glad to see them again. 

Still, while I enjoy RegEx immensely, I had troubles with both calculations and further integrating them into a shell environment. In hindsight, I'm not sure why exactly I had difficulties, since the quizes didn't require too much besides RegEx. 

An example of data processing can be seen here, where I calculate the probabiltity of seeing the word 'honey' in a sentence in the book Life of Bee. Here we find all sentences which contain the word "honey." By using \b around the word, we exclude all inflected forms of the word:

`cat life_of_bee.sent | grep -E "\bhoney\b"  | wc -l` 

Then, we divide the result with the total number of sentences, which is an amount we get with:

`wc -l life_of_bee.sent`

To note, the .sent file is the text of the book converted to a sentence per line format. 


## Week 4: Scripting and UNIX Configuration Files

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


## Week 5: Installing and Running Programs

For the second to last week I learned about commands such as make. Additionally, while I already had some familarity with Python from another course, I felt like I got a deeper understanding about what exactly pip is.

One of the Python modules I installed this week was the BLLIP parser. After importing the module in python3, we can use the following command to parse the sentence "NLP is fun!":

rrp.simple_parse("NLP is fun!")

This gives us the following parse tree:

'(S1 (S (NP (NNP NLP)) (VP (VBZ is) (NP (NN fun))) (. !)))'

Here is a more visually appealing presentation of the syntax tree.

![Syntax Tree](https://i.imgur.com/tHZb1ZZ.png)

## Week 6: Version Control

During this last week, I got to prepare for the final assignment of the course (which basically is this page). I have visited github before, and occasionally downloaded files from github, but I never really knew what it was, or how version control works. 
