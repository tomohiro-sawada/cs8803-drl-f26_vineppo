<!--Transcribed from: https://www.pair.toronto.edu/cs8803-drl-f26/project.html-->

[CS 8803 - DRL - F2026](/cs8803-drl-f26/)

*   [Home](/cs8803-drl-f26/)
*   [Paper Presentation](/cs8803-drl-f26/project.html)
*   [Schedule](/cs8803-drl-f26/schedule.html)
*   [Staff](/cs8803-drl-f26/staff.html)

*   [Syllabus](https://www.pairlab.github.io/cs8803-drl-f26/assets/CS-8803-DRL-fall-2026-syllabus.pdf)

This site uses [Just the Docs](https://github.com/just-the-docs/just-the-docs), a documentation theme for Jekyll.

[](#paper-presentation-and-final-project)Paper Presentation and Final Project
=============================================================================

The purpose of this segment is to deepen your understanding and engagement with cutting-edge RL research through the detailed study, presentation, and implementation of a significant paper in the field. This exercise aims to emulate the process of a conference presentation combined with a critical review session.

Your [team assignments, papers, and deadlines are here](https://docs.google.com/spreadsheets/d/1aLjzTaep6zFwFXzyeeD0YCzK9Ttj8O5ndPJQ-6uC1q0/edit). Please check your assignment and contact the instructors about any errors.

[](#overview)Overview
---------------------

Each team prepares **one 12–15 minute presentation video** on its assigned paper and revises it over the semester, then builds on the paper for the final project.

Stage

When

What to submit

**1\. Initial draft**

One week before your assigned date

Slides (PDF) and video script

**2\. Revised video + quiz**

Your assigned date

Video uploaded to Media Gallery, plus 5 quiz questions

**3\. Final project + final video**

**Monday, December 7**

Code, report, final edited video, team-contribution survey

Only **one member per team** needs to submit on Canvas. The submission counts for the whole team.

[](#schedule)Schedule
---------------------

Videos are released to the class on Tuesdays and Thursdays, four teams per lecture (three in the final batch). The **first initial drafts are due Thursday, October 22.**

Draft due (slides + script)

Video + quiz due / release

Teams

Lecture topic

Thu, Oct 22

Thu, Oct 29

T01–T04

Critic-free policy gradients

Tue, Oct 27

Tue, Nov 3

T05–T08

Learning on hardware and post-training pretrained policies

Thu, Oct 29

Thu, Nov 5

T09–T12

Credit assignment under long horizons

Tue, Nov 3

Tue, Nov 10

T13–T16

Search and test-time compute

Thu, Nov 5

Thu, Nov 12

T17–T20

Expert iteration and verifier-bounded self-improvement

Tue, Nov 10

Tue, Nov 17

T21–T24

Reward hacking, over-optimization, and constrained RL

Thu, Nov 12

Thu, Nov 19

T25–T28

Verification and environment design

Tue, Nov 17

Tue, Nov 24

T29–T32

Multi-agent RL and coordination

Thu, Nov 19

Thu, Nov 26

T33–T36

Agentic RL as a systems problem

Tue, Nov 24

Tue, Dec 1

T37–T39

Reasoning models and the elicitation debate

**Mon, Dec 7**

—

All teams

Final project and final video

The spreadsheet is the source of truth. If anything here disagrees with it, follow the spreadsheet.

[](#submissions)Submissions
---------------------------

### [](#1-initial-draft-one-week-before-your-assigned-date)1\. Initial Draft (one week before your assigned date)

*   Submit to the Canvas assignment **“Paper Presentation – Initial Draft.”**
*   Submit your **slides (PDF)** and your **video script**. No video is required yet.
*   The instructors will give feedback before you record.

### [](#2-revised-video-and-quiz-questions-your-assigned-date)2\. Revised Video and Quiz Questions (your assigned date)

*   Upload your 12–15 minute video to **Media Gallery** in Canvas (left course navigation → **\+ Add Media → Media Upload**).
*   Title it `Team [#] – [Paper Title]`.
*   Submit to the Canvas assignment **“Paper Presentation – Revised Video”** with:
    *   the link to your Media Gallery video
    *   your 5 quiz questions with answers
*   Uploads stay hidden until they are released to the class on your date, so uploading early is fine. Leave time for processing, since longer videos can take a while.

### [](#3-final-video-with-the-final-project-december-7)3\. Final Video (with the final project, December 7)

*   Upload the final edited version of the same video to Media Gallery, titled `Team [#] – [Paper Title] – FINAL`.
*   Update it to incorporate feedback and include your implementation, modification, and findings.
*   Include the link in your final report and Canvas submission.
*   **Only this final version is graded for the video component.**

[](#presentation-structure)Presentation Structure
-------------------------------------------------

### [](#format)Format

*   **Duration**: 12–15 minutes.
    *   About **5 minutes** of background and context.
    *   About **7–10 minutes** on the main material of the paper.
*   **Slides**: We strongly urge you to follow the design and layout of the professor’s lecture slides, so every presentation is consistent and easy to follow.
*   **Analysis**: Cover the paper’s background, problem statement, methodology, experiments, key findings, and implications. Highlight the significance of the work and its contributions to the field. Organize your deck around these sections:
    *   **Background**: Discuss prior works leading up to the publication of the paper.
    *   **Problem Statement**: Clearly outline the primary problem the paper addresses.
    *   **Motivation**: Discuss why the problem is important and what drives the research.
    *   **Method**: Summarize the approach or methodology the paper proposes, highlighting any novel techniques or theories.
    *   **Experiments**: Provide a short analysis and comment on the experiments conducted in the paper, focusing on their design, execution, and the validity of the conclusions drawn from them.
    *   **Future Work**: Discuss works that have resulted from or built upon the paper.
    *   **Critique**: Offer a critical analysis of the paper’s strengths, weaknesses, impact, and contributions. Anticipate potential criticisms and questions that might arise, simulating the authors’ rebuttal to peer review comments.

### [](#quiz-questions)Quiz Questions

*   Prepare **5 quiz questions with answers** that check whether the audience has grasped the key concepts of the paper. They should be answerable based solely on your presentation.
*   Questions must be Canvas-friendly and auto-gradable (e.g., multiple choice, true/false).
*   The full set should take no more than 5 minutes to answer and focus on the key takeaways.
*   Submit them with your revised video on your assigned date.

### [](#discussion)Discussion

*   All students are expected to watch the released presentations.
*   Leave questions and feedback in the Media Gallery comments on each video.
*   Presenting teams should respond to questions on their own video.

[](#final-project-due-december-7)Final Project (due December 7)
---------------------------------------------------------------

### [](#purpose)Purpose

Reproduce the main results of your paper, then implement a meaningful modification or extension (e.g., a new environment or an algorithmic improvement), run experiments, and report your findings. This is your opportunity to apply what you’ve learned, showcase creative ideas, and demonstrate a strong understanding of your chosen method.

Your grade does **not** depend on your idea outperforming the baseline. We do expect a proper analysis either way: why did your modification succeed or fail? Provide supporting evidence, as you would in a short conference paper.

If implementing the paper is impractical (e.g., it requires resources beyond the scope of the course), you **must inform the instructors as early as possible**, with a detailed justification and a proposed alternative.

### [](#deliverables)Deliverables

**1\. Code**

*   Submit as a GitHub repository link (included in your report) or a ZIP file uploaded with your submission.
*   Include a README with setup and run instructions, dependencies, and the scripts needed to reproduce your experiments.
*   The code should be self-contained. A Python notebook that walks through the main points, with math and code side by side, is recommended.

**2\. Final Report (PDF, about 2–4 pages, not a strict limit)**

*   **Introduction and Background**: Briefly summarize the core idea of the paper and your reproduction.
*   **Modification Description**: Clearly describe the modification you implemented and the motivation behind it.
*   **Experimental Setup**: Describe your evaluation strategy, including environments or datasets, training setup, and evaluation metrics.
*   **Results and Discussion**: Include graphs, tables, or visuals comparing your modification to the original method, and discuss your observations.
*   **Video Link**: Link to your final edited presentation video.

**3\. Final Video**

*   The final edited version of your 12–15 minute presentation video (see Submission 3 above).

**4\. Team Contribution Survey**

*   Each team member individually fills out a survey describing each member’s contribution (e.g., implementation, experimentation, writing).

[](#general-guidelines)General Guidelines
-----------------------------------------

*   **Engagement**: Engage with your classmates’ questions and feedback throughout the semester, as you would at a real conference.
*   **Preparation**: In addition to your slides, consider preparing additional materials if they help clarify complex concepts in the paper.
*   **Feedback**: You will receive feedback from instructors on your draft and from peers on your released video. Use it to refine your final video and project.

* * *

This site uses [Just the Docs](https://github.com/just-the-docs/just-the-docs), a documentation theme for Jekyll.
