# Mobile Cloud Task Scheduler

Python implementation of the energy-aware task scheduling algorithm from Lin et al.'s 2014 IEEE Cloud Computing paper: "Energy and Performance-Aware Task Scheduling in a Mobile Cloud Computing Environment."

## Overview

This project implements the MCC (Mobile Cloud Computing) task scheduling algorithm that optimizes energy consumption in mobile devices while satisfying hard application completion time constraints. The algorithm addresses the problem of scheduling task graphs across heterogeneous local cores and cloud resources.

## Problem Statement

Given a directed acyclic graph (DAG) of tasks, the algorithm determines:
1. Which tasks to offload to the cloud
2. How to map remaining tasks onto heterogeneous cores in the mobile device
3. How to schedule tasks on local cores and wireless communication channels

**Objective**: Minimize total energy consumption while satisfying task precedence requirements and application completion time constraints.

## Algorithm Approach

### Phase 1: Initial Scheduling
Generates minimal-delay scheduling to minimize application completion time:
- Primary assignment based on execution time comparison
- Task prioritizing using critical path analysis
- Execution unit selection with ready time calculations

### Phase 2: Task Migration
Reduces energy consumption through iterative task migration:
- Evaluates migration candidates (local core to cloud, local core to local core)
- Uses linear-time O(N) rescheduling algorithm to maintain precedence constraints
- Ensures final schedule satisfies completion time constraint

## Implementation Details

**Language**: Python  
**Complexity**: O(E × K) where E = edges in task graph, K = number of cores  
**Key Components**:
- DAG precedence enforcement
- Priority computation for heterogeneous resources
- Constraint-preserving task migration
- Energy and completion time tracking

## Test Cases

Implementation validated across 5 increasingly complex scenarios:

1. **Example 1**: 10-task DAG from paper's Figure 1 (single entry/exit)
2. **Example 2**: 10-task DAG with modified topology
3. **Example 3**: 20-task DAG with extended execution time table
4. **Example 4**: 20-task DAG with multiple entry tasks
5. **Example 5**: 20-task DAG with multiple entry and exit tasks

Each test case demonstrates energy reduction while maintaining completion time constraints.

## Configuration

Default system parameters:
- **Local cores**: 3 heterogeneous cores with power consumption P1=1W, P2=2W, P3=4W
- **Cloud parameters**: T_send=3, T_cloud=1, T_receive=1
- **Wireless power**: P_s=0.5W
- **Task migration constraint**: T_max ≈ 1.5 × T_total_initial


Outputs for each test case:
- Initial scheduling results (time-optimized)
- Final scheduling results (energy-optimized)
- Energy breakdown by core and cloud
- Total completion times

## Reference
Lin, X., Wang, Y., Xie, Q., & Pedram, M. (2014). Energy and Performance-Aware Task Scheduling in a Mobile Cloud Computing Environment. *2014 IEEE International Conference on Cloud Computing*, 192-199.

## Key Results

The algorithm successfully:
- Validates against paper's Figure 1 example
- Handles complex task graphs with up to 20 tasks
- Supports multiple entry/exit tasks
- Maintains linear-time rescheduling complexity
- Achieves energy reduction while respecting time constraints
