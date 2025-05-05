SUMMARIZE_PROMPT = """
### **Task Description**

Given a mathematical question and its detailed solution, the task is to segment the solution into high-level problem-solving phases. The goal is to group consecutive steps into meaningful phases and output only the start and end steps of each phase.

Note: Each **reasoning step** in the solution is separated by a **double line break ("\n\n")**.

------

### **Requirements**

1. **Segment the full solution into distinct problem-solving phases** based on logical progression.
2. **Each phase should have a start and an end step**.
3. **A phase can appear multiple times** in different parts of the solution.
4. **The order of phases is flexible**—they can appear in any logical sequence depending on the nature of the solution.
5. **Only the first and last steps of each phase should be output**, reducing redundancy.

------

### **High-Level Problem-Solving Phases**

Each step in the solution should belong to one of the following **ten high-level phases**:

1. **Understanding the Problem**: Identifying given data, definitions, and the goal.
2. **Reformulating the Problem**: Changing variables, rewriting expressions, or restructuring sums.
3. **Applying Known Theorems/Properties**: Using standard formulas, identities, or mathematical principles.
4. **Breaking Down into Subproblems**: Decomposing the problem into manageable components.
5. **Computing or Simplifying Expressions**: Performing algebraic manipulation or numerical evaluation.
6. **Substituting Known Values or Results**: Using precomputed values or standard mathematical constants.
7. **Reassess and Verify Local Steps**: Checking for errors or inconsistencies within a small part of the solution.
8. **Reassess the Whole Solution**: Reviewing the entire solution for logical correctness and consistency.
9. **Exploring Alternative Approaches**: Considering different methods to solve the problem.
10. **Finalize and Present the Answer**: Writing the final result and ensuring clarity.

------

### **Output Format**

The output should consist of **multiple phases**, each represented in the following format:

```
[Phase X]: {Phase Name}
[Start]: {Text of first step in the phase}
[End]: {Text of last step in the phase}
```

Where:

- **Phase X** represents the index of the phase (e.g., Phase 1, Phase 2, etc.).
- **Phase Name** is one of the ten high-level categories.
- **Start** is the first step of the phase.
- **End** is the last step of the phase.

#### **Example Output**

```
[Phase 1]: Understanding the Problem
[Start]: [Text of step 1]
[End]: [Text of step 3]

[Phase 2]: Reformulating the Problem
[Start]: [Text of step 4]
[End]: [Text of step 5]

[Phase 3]: Computing or Simplifying Expressions
[Start]: [Text of step X]
[End]: [Text of step Y]
...

[Phase 4]: Finalize and Present the Answer
[Start]: [Text of step X]
[End]: [Text of step Y]
```""".strip()

