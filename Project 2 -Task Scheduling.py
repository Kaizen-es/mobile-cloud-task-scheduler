#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Project 2 - Task Scheduling
def build_graph(graph):
    pred = {i: [] for i in graph.keys()}
    for task, successors in graph.items():
        for succ in successors:
            pred[succ].append(task)
    return pred

def compute_priority(graph, w):
    # Compute priority for each task using equation (15) from paper
    priority = {}
    
    def calc(task):
        if task in priority:
            return priority[task]
        
        if len(graph[task]) == 0:
            priority[task] = w[task]
        else:
            max_succ = max(calc(succ) for succ in graph[task])
            priority[task] = w[task] + max_succ
        
        return priority[task]
    
    # Compute for all tasks
    for task in graph.keys():
        calc(task)
    
    return priority


def ready_time_ex(task, pred, schedule):
    """
    Ready time RT^ex for executing task on a LOCAL core from 
    equation (3)
    """
    if not pred[task]:
        return 0
    
    max_time = 0
    for p in pred[task]:
        if schedule['assignment'][p] == 0:  # predecessor on cloud
            max_time = max(max_time, schedule['FT_wr'][p])
        else:  # predecessor on local core
            max_time = max(max_time, schedule['FT_ex'][p])
    return max_time


def ready_time_ws(task, pred, schedule):
    """
    Ready time RT^ws for sending a task to cloud from
    equation (4)
    """
    if not pred[task]:
        return 0
    
    max_time = 0
    for p in pred[task]:
        if schedule['assignment'][p] == 0:  # predecessor on cloud
            max_time = max(max_time, schedule['FT_ws'][p])
        else:  # predecessor on local core
            max_time = max(max_time, schedule['FT_ex'][p])
    return max_time


def ready_time_c(task, pred, schedule):
    """
    Ready time component for cloud execution from
    Equation (5)
    """
    if not pred[task]:
        return 0
    
    max_time = 0
    for p in pred[task]:
        if schedule['assignment'][p] == 0:  
            max_time = max(max_time, schedule['FT_c'][p])
    return max_time

def energy(schedule, exec_times, P, P_s, T_s):

    # Total energy consumption from equations (7-9).
    E_cores = {1: 0, 2: 0, 3: 0}
    E_cloud = 0
    
    for task, loc in schedule['assignment'].items():
        if loc == 0:  # cloud task
            E_cloud += P_s * T_s
        else:  # local task
            T_ex = exec_times[task][loc - 1]
            E_cores[loc] += P[loc - 1] * T_ex
    
    E_total = sum(E_cores.values()) + E_cloud
    return E_total, E_cores, E_cloud


def T_total(schedule):
    # Application completion time T^total from equation (10)
    
    max_time = 0
    for task, loc in schedule['assignment'].items():
        if loc == 0:  
            ft = schedule.get('FT_wr', {}).get(task, 0)
        else: 
            ft = schedule.get('FT_ex', {}).get(task, 0)
        max_time = max(max_time, ft)
    return max_time


def copy_schedule(schedule):
    return {
        'assignment': dict(schedule['assignment']),
        'FT_ex': dict(schedule['FT_ex']),
        'FT_ws': dict(schedule['FT_ws']),
        'FT_c': dict(schedule['FT_c']),
        'FT_wr': dict(schedule['FT_wr']),
        'ST': dict(schedule['ST'])
    }


def initial_scheduling(graph, exec_times, T_s, T_c, T_r, P, P_s, K):
    # Initial scheduling minimizes application completion time T^total.
    
    N = len(graph)
    pred = build_graph(graph)
    
    # Phase 1: Primary assignment
    T_re = T_s + T_c + T_r  # remote execution time 
    cloud_tasks = set()
    
    for i in range(1, N + 1):
        T_local_min = min(exec_times[i]) 
        if T_re < T_local_min:
            cloud_tasks.add(i)
            
    # Phase 2: Task Prioritizing
    w = {}
    for i in range(1, N + 1):
        if i in cloud_tasks:
            w[i] = T_re
        else:
            w[i] = sum(exec_times[i]) / K
    
    priority = compute_priority(graph, w)
    
    # Phase 3: Execution Unit Selection
    
    # Sort tasks by priority (highest first)
    sorted_tasks = sorted(range(1, N + 1), key=lambda x: priority[x], reverse=True)
    
    # Initialize schedule data structure
    schedule = {
        'assignment': {},  
        'FT_ex': {},       # finish time for local execution
        'FT_ws': {},       # finish time for wireless send
        'FT_c': {},        # finish time for cloud compute
        'FT_wr': {},       # finish time for wireless receive
        'ST': {}           # start time
    }
    
    # Track resource availability
    core_available = {1: 0, 2: 0, 3: 0}
    wireless_available = 0
    
    for task in sorted_tasks:
        # Check if task is a cloud task from Phase 1
        if task in cloud_tasks:
            # Cloud task must be assigned to cloud
            RT_ws = ready_time_ws(task, pred, schedule)
            ST_ws = max(RT_ws, wireless_available)
            FT_ws = ST_ws + T_s
            RT_c = max(FT_ws, ready_time_c(task, pred, schedule))
            FT_c = RT_c + T_c
            FT_wr = FT_c + T_r
            
            schedule['assignment'][task] = 0
            schedule['ST'][task] = ST_ws
            schedule['FT_ws'][task] = FT_ws
            schedule['FT_c'][task] = FT_c
            schedule['FT_wr'][task] = FT_wr
            wireless_available = FT_ws
        else:
            # Compare local vs cloud and pick minimum finish time
            # finish time for cloud
            RT_ws = ready_time_ws(task, pred, schedule)
            ST_ws = max(RT_ws, wireless_available)
            FT_ws = ST_ws + T_s
            RT_c = max(FT_ws, ready_time_c(task, pred, schedule))
            FT_c = RT_c + T_c
            FT_wr = FT_c + T_r
            cloud_finish = FT_wr
            
            # finish time for each local core
            RT_ex = ready_time_ex(task, pred, schedule)
            best_core = None
            best_core_finish = float('inf')
            
            for k in range(1, K + 1):
                ST_ex = max(RT_ex, core_available[k])
                FT_ex = ST_ex + exec_times[task][k - 1]
                if FT_ex < best_core_finish:
                    best_core_finish = FT_ex
                    best_core = k
            
            # Choose location with minimum finish time
            if cloud_finish < best_core_finish:
                # Assign to cloud
                schedule['assignment'][task] = 0
                schedule['ST'][task] = ST_ws
                schedule['FT_ws'][task] = FT_ws
                schedule['FT_c'][task] = FT_c
                schedule['FT_wr'][task] = FT_wr
                wireless_available = FT_ws
            else:
                # Assign to best local core
                schedule['assignment'][task] = best_core
                ST_ex = max(RT_ex, core_available[best_core])
                FT_ex = ST_ex + exec_times[task][best_core - 1]
                schedule['ST'][task] = ST_ex
                schedule['FT_ex'][task] = FT_ex
                core_available[best_core] = FT_ex
    
    return schedule, priority


# TASK MIGRATION 

def task_migration(graph, exec_times, T_s, T_c, T_r, P, P_s, K, initial_schedule, T_max):
 
    # Task Migration minimizes energy while keeping T^total <= T^max.

    pred = build_graph(graph)
    current_schedule = copy_schedule(initial_schedule)
    iteration_log = []
    iteration = 0
    
    while True:
        iteration += 1
        current_T = T_total(current_schedule)
        current_E, _, _ = energy(current_schedule, exec_times, P, P_s, T_s)
        
        # Find tasks on local cores 
        local_tasks = [t for t, loc in current_schedule['assignment'].items() if loc != 0]
        
        if not local_tasks:
            break  # No tasks to migrate
        
        # Track best migration choice
        best_choice = None
        best_energy_reduction = 0
        best_ratio = 0
        best_new_schedule = None
        no_time_increase_found = False
        
        # Try all N' × K migration choices
        for task in local_tasks:
            k_ori = current_schedule['assignment'][task]
            
            # Target locations: other cores (1,2,3 except current) and cloud (0)
            for k_tar in range(0, K + 1):
                if k_tar == k_ori:
                    continue  # Skip same location
                
                # Run kernel algorithm to get new schedule
                new_schedule = kernel_algorithm(
                    graph, exec_times, T_s, T_c, T_r, K,
                    current_schedule, task, k_tar, pred
                )
                
                new_T = T_total(new_schedule)
                new_E, _, _ = energy(new_schedule, exec_times, P, P_s, T_s)
                
                energy_reduction = current_E - new_E
                time_increase = new_T - current_T
                
                # Skip if energy not reduced
                if energy_reduction <= 0:
                    continue
                
                # Case 1: Energy reduced and time not increased
                if time_increase <= 0:
                    if energy_reduction > best_energy_reduction or not no_time_increase_found:
                        best_energy_reduction = energy_reduction
                        best_choice = (task, k_ori, k_tar)
                        best_new_schedule = new_schedule
                        no_time_increase_found = True
                
                # Case 2: Energy reduced but time increased 
                elif new_T <= T_max and not no_time_increase_found:
                    ratio = energy_reduction / time_increase
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_choice = (task, k_ori, k_tar)
                        best_new_schedule = new_schedule
        
        # Check if any valid migration found
        if best_choice is None:
            break
        
        # Apply the migration
        task, k_ori, k_tar = best_choice
        current_schedule = best_new_schedule
        
        new_T = T_total(current_schedule)
        new_E, _, _ = energy(current_schedule, exec_times, P, P_s, T_s)
        
        # Log the migration
        loc_ori = f"Core{k_ori}" if k_ori > 0 else "Cloud"
        loc_tar = f"Core{k_tar}" if k_tar > 0 else "Cloud"
        
        iteration_log.append({
            'iteration': iteration,
            'task': task,
            'from': loc_ori,
            'to': loc_tar,
            'T_total': new_T,
            'E_total': new_E
        })
        
        # Safety limit
        if iteration > 100:
            break
    
    return current_schedule, iteration_log


def kernel_algorithm(graph, exec_times, T_s, T_c, T_r, K, schedule, v_tar, k_tar, pred):
 
    # Kernel Algorithm - Linear-time rescheduling after task migration.
  
    N = len(graph)
    k_ori = schedule['assignment'][v_tar]
    
    # Build sequence sets from current schedule
    sequences = {k: [] for k in range(0, K + 1)}
    
    for task in range(1, N + 1):
        loc = schedule['assignment'][task]
        st = schedule['ST'][task]
        sequences[loc].append((st, task))
    
    # Sort by start time and extract just task ids
    for k in sequences:
        sequences[k].sort()
        sequences[k] = [t for (st, t) in sequences[k]]
    
    # Remove v_tar from original sequence
    if v_tar in sequences[k_ori]:
        sequences[k_ori].remove(v_tar)
    
    # Insert v_tar into target sequence at proper position
    if k_tar > 0:
        RT_tar = ready_time_ex(v_tar, pred, schedule)
    else:
        RT_tar = ready_time_ws(v_tar, pred, schedule)
    
    # Find insertion position: after tasks with ST < RT_tar
    insert_pos = 0
    for i, task in enumerate(sequences[k_tar]):
        if schedule['ST'][task] < RT_tar:
            insert_pos = i + 1
        else:
            break
    sequences[k_tar].insert(insert_pos, v_tar)
    
    # Create new assignment and reschedule
    new_assignment = dict(schedule['assignment'])
    new_assignment[v_tar] = k_tar
    
    new_schedule = reschedule_with_sequences(
        graph, exec_times, T_s, T_c, T_r, K, sequences, new_assignment, pred
    )
    
    return new_schedule

def reschedule_with_sequences(graph, exec_times, T_s, T_c, T_r, K, sequences, assignment, pred):
    # Reschedule all tasks given fixed sequences and assignments.
    N = len(graph)
    
    # Initialize new schedule
    new_schedule = {
        'assignment': assignment,
        'FT_ex': {},
        'FT_ws': {},
        'FT_c': {},
        'FT_wr': {},
        'ST': {}
    }
    
    # ready1[i] = count of unscheduled predecessors
    ready1 = {task: len(pred[task]) for task in range(1, N + 1)}
    
    # Track position in each sequence 
    seq_pos = {k: 0 for k in range(0, K + 1)}
    
    # Resource availability
    core_available = {1: 0, 2: 0, 3: 0}
    wireless_available = 0
    
    # Initialize stack with tasks that are ready
    stack = []
    scheduled = set()
    
    def is_ready(task):
        # Check if task is ready to be scheduled
        if ready1[task] > 0:
            return False
        loc = assignment[task]
        if seq_pos[loc] >= len(sequences[loc]):
            return False
        return sequences[loc][seq_pos[loc]] == task
    
    # Find initially ready tasks
    for task in range(1, N + 1):
        if is_ready(task):
            stack.append(task)
    
    while stack:
        task = stack.pop()
        
        if task in scheduled:
            continue
        scheduled.add(task)
        
        loc = assignment[task]
        
        # Schedule the task
        if loc == 0:
            # Cloud execution
            RT_ws = ready_time_ws_partial(task, pred, new_schedule, assignment)
            ST_ws = max(RT_ws, wireless_available)
            FT_ws = ST_ws + T_s
            RT_c = max(FT_ws, ready_time_c_partial(task, pred, new_schedule, assignment))
            FT_c = RT_c + T_c
            FT_wr = FT_c + T_r
            
            new_schedule['ST'][task] = ST_ws
            new_schedule['FT_ws'][task] = FT_ws
            new_schedule['FT_c'][task] = FT_c
            new_schedule['FT_wr'][task] = FT_wr
            wireless_available = FT_ws
        else:
            # Local core execution
            RT_ex = ready_time_ex_partial(task, pred, new_schedule, assignment)
            ST_ex = max(RT_ex, core_available[loc])
            FT_ex = ST_ex + exec_times[task][loc - 1]
            
            new_schedule['ST'][task] = ST_ex
            new_schedule['FT_ex'][task] = FT_ex
            core_available[loc] = FT_ex
        
        # Advance sequence position for this location
        seq_pos[loc] += 1
        
        # Update ready1 for successors
        for succ in graph[task]:
            ready1[succ] -= 1
        
        # Check for newly ready tasks
        for t in range(1, N + 1):
            if t not in scheduled and is_ready(t):
                stack.append(t)
    
    return new_schedule


def ready_time_ex_partial(task, pred, schedule, assignment):
    # Compute RT^ex using partially filled schedule
    if not pred[task]:
        return 0
    max_time = 0
    for p in pred[task]:
        if assignment[p] == 0:
            max_time = max(max_time, schedule.get('FT_wr', {}).get(p, 0))
        else:
            max_time = max(max_time, schedule.get('FT_ex', {}).get(p, 0))
    return max_time


def ready_time_ws_partial(task, pred, schedule, assignment):
    # Compute RT^ws using partially filled schedule
    if not pred[task]:
        return 0
    max_time = 0
    for p in pred[task]:
        if assignment[p] == 0:
            max_time = max(max_time, schedule.get('FT_ws', {}).get(p, 0))
        else:
            max_time = max(max_time, schedule.get('FT_ex', {}).get(p, 0))
    return max_time


def ready_time_c_partial(task, pred, schedule, assignment):
    # Compute cloud ready time component using partially filled schedule
    if not pred[task]:
        return 0
    max_time = 0
    for p in pred[task]:
        if assignment[p] == 0:
            max_time = max(max_time, schedule.get('FT_c', {}).get(p, 0))
    return max_time

def run_example(example_num, graph, exec_times, T_s, T_c, T_r, P, P_s, K):
    
    N = len(graph)
    
    print(f"INPUT EXAMPLE {example_num}")
    print("Task Graph Edges:")
    for task, succs in graph.items():
        if succs:
            print(f"  v{task} -> {['v' + str(s) for s in succs]}")
    print()
    
    print("Execution Time Table:")
    print("  Task | Core1 | Core2 | Core3")
    print("  " + "-" * 28)
    for task in range(1, N + 1):
        t = exec_times[task]
        print(f"  v{task:2d}  |   {t[0]}   |   {t[1]}   |   {t[2]}")
    print()
    
    print(f"Cloud: T_s={T_s}, T_c={T_c}, T_r={T_r}")
    print(f"Power: P1={P[0]}, P2={P[1]}, P3={P[2]}, P_s={P_s}")
    print()
    
    # STEP 1: Initial Scheduling
    print("INITIAL SCHEDULING")
    
    schedule, priority = initial_scheduling(graph, exec_times, T_s, T_c, T_r, P, P_s, K)
    
    print("Task Priorities:")
    for task in range(1, N + 1):
        print(f"  priority(v{task:2d}) = {priority[task]:.2f}")
    print()
    
    print("Initial Schedule:")
    print("  Task | Location | Start | Finish")
    print("  " + "-" * 35)
    for task in range(1, N + 1):
        loc = schedule['assignment'][task]
        loc_str = f"Core{loc}  " if loc > 0 else "Cloud  "
        st = schedule['ST'][task]
        ft = schedule['FT_wr'][task] if loc == 0 else schedule['FT_ex'][task]
        print(f"  v{task:2d}  | {loc_str}  |  {st:4.0f} |  {ft:4.0f}")
    print()
    
    T_initial = T_total(schedule)
    E_initial, E_cores_init, E_cloud_init = energy(schedule, exec_times, P, P_s, T_s)
    
    print(f"Results:")
    print(f"  T^total = {T_initial}")
    print(f"  E^total = {E_initial}")
    print(f"  E1={E_cores_init[1]}, E2={E_cores_init[2]}, E3={E_cores_init[3]}, E_cloud={E_cloud_init}")
    print()

    print("Manual Energy Calculation (Initial Schedule):")
    for task in range(1, N + 1):
        loc = schedule['assignment'][task]
        if loc == 0:
            E_i = P_s * T_s
            print(f"  E_{task:2d} = P_s × T_s = {P_s} × {T_s} = {E_i}")
        else:
            T_ex = exec_times[task][loc - 1]
            E_i = P[loc - 1] * T_ex
            print(f"  E_{task:2d} = P{loc} × T_{task},{loc} = {P[loc-1]} × {T_ex} = {E_i}")
    print()
    print(f"  E^total = E1 + E2 + E3 + E_cloud")
    print(f"  E^total = {E_cores_init[1]} + {E_cores_init[2]} + {E_cores_init[3]} + {E_cloud_init} = {E_initial}")
    print()
    
    T_max = 1.5 * T_initial
    
    # STEP 2: Task Migration
    print("TASK MIGRATION")
    print(f"T^max = 1.5 × {T_initial} = {T_max}")

    final_schedule, log = task_migration(
        graph, exec_times, T_s, T_c, T_r, P, P_s, K, schedule, T_max
    )
    
    print("Migration Steps:")
    if log:
        for entry in log:
            print(f"  Step {entry['iteration']}: v{entry['task']} moved from {entry['from']} to {entry['to']}")
            print(f"    -> T^total={entry['T_total']}, E^total={entry['E_total']}")
    else:
        print("No migrations needed")
    print()
    
    print("Final Schedule:")
    print("  Task | Location | Start | Finish")
    print("  " + "-" * 35)
    for task in range(1, N + 1):
        loc = final_schedule['assignment'][task]
        loc_str = f"Core{loc}  " if loc > 0 else "Cloud  "
        st = final_schedule['ST'][task]
        ft = final_schedule['FT_wr'][task] if loc == 0 else final_schedule['FT_ex'][task]
        print(f"  v{task:2d}  | {loc_str}  |  {st:4.0f} |  {ft:4.0f}")
    print()
    
    T_final = T_total(final_schedule)
    E_final, E_cores_final, E_cloud_final = energy(final_schedule, exec_times, P, P_s, T_s)
    
        
    print(f"Results:")
    print(f"  T^total = {T_final}")
    print(f"  E^total = {E_final}")
    print(f"  E1={E_cores_final[1]}, E2={E_cores_final[2]}, E3={E_cores_final[3]}, E_cloud={E_cloud_final}")
    print()
    
    print("Manual Energy Calculation (Final Schedule):")
    for task in range(1, N + 1):
        loc = final_schedule['assignment'][task]
        if loc == 0:
            E_i = P_s * T_s
            print(f"  E_{task:2d} = P_s × T_s = {P_s} × {T_s} = {E_i}")
        else:
            T_ex = exec_times[task][loc - 1]
            E_i = P[loc - 1] * T_ex
            print(f"  E_{task:2d} = P{loc} × T_{task},{loc} = {P[loc-1]} × {T_ex} = {E_i}")
    print()
    print(f"  E^total = E1 + E2 + E3 + E_cloud")
    print(f"  E^total = {E_cores_final[1]} + {E_cores_final[2]} + {E_cores_final[3]} + {E_cloud_final} = {E_final}")
    print()

    print("SUMMARY")
    print(f"  {'Metric':<15} | {'Initial':>12} | {'Final':>12}")
    print("  " + "-" * 45)
    print(f"  {'T^total':<15} | {T_initial:>12} | {T_final:>12}")
    print(f"  {'E^total':<15} | {E_initial:>12} | {E_final:>12}")
    print(f"  {'E1':<15} | {E_cores_init[1]:>12} | {E_cores_final[1]:>12}")
    print(f"  {'E2':<15} | {E_cores_init[2]:>12} | {E_cores_final[2]:>12}")
    print(f"  {'E3':<15} | {E_cores_init[3]:>12} | {E_cores_final[3]:>12}")
    print(f"  {'E_cloud':<15} | {E_cloud_init:>12} | {E_cloud_final:>12}")
    print()

print("PROJECT 2: TASK SCHEDULING")

# System parameters
T_s = 3
T_c = 1
T_r = 1
P = [1, 2, 4]
P_s = 0.5
K = 3

# EXAMPLE 1: Figure 1 from the paper
graph_1 = {
    1: [2, 3, 4, 5, 6],
    2: [8, 9],
    3: [7],
    4: [8, 9],
    5: [9],
    6: [8],
    7: [10],
    8: [10],
    9: [10],
    10: []
}

exec_times_1 = {
    1: [9, 7, 5],
    2: [8, 6, 5],
    3: [6, 5, 4],
    4: [7, 5, 3],
    5: [5, 4, 2],
    6: [7, 6, 4],
    7: [8, 5, 3],
    8: [6, 4, 2],
    9: [5, 3, 2],
    10: [7, 4, 2]
}

run_example(1, graph_1, exec_times_1, T_s, T_c, T_r, P, P_s, K)

# EXAMPLE 2: Same number of tasks, but different connections

graph_2 = {
    1: [2, 3, 4, 5, 6],
    2: [7],
    3: [7],
    4: [7, 8, 9],
    5: [9],
    6: [9],
    7: [10],
    8: [10],
    9: [10],
    10: []
}

exec_times_2 = exec_times_1

run_example(2, graph_2, exec_times_2, T_s, T_c, T_r, P, P_s, K)

# EXAMPLE 3: 20 tasks
graph_3 = {
    1: [2, 3, 4, 5, 6],
    2: [7],
    3: [7],
    4: [7, 8, 9],
    5: [9],
    6: [9],
    7: [13],
    8: [10, 12, 14],
    9: [11],
    10: [15],
    11: [15, 16],
    12: [16, 17],
    13: [17, 18],
    14: [18],
    15: [19],
    16: [19],
    17: [19],
    18: [19],
    19: [20],
    20: []
}

exec_times_3 = {
    1: [9, 7, 5],
    2: [8, 6, 5],
    3: [6, 5, 4],
    4: [7, 5, 3],
    5: [5, 4, 2],
    6: [7, 6, 4],
    7: [8, 5, 3],
    8: [6, 4, 2],
    9: [5, 3, 2],
    10: [7, 4, 2],
    11: [6, 4, 3],
    12: [5, 4, 2],
    13: [3, 2, 1],
    14: [6, 4, 2],
    15: [4, 3, 2],
    16: [6, 3, 2],
    17: [5, 4, 3],
    18: [5, 4, 1],
    19: [4, 3, 1],
    20: [6, 3, 2]
}

run_example(3, graph_3, exec_times_3, T_s, T_c, T_r, P, P_s, K)

# EXAMPLE 4: 20 tasks with multiple entry tasks
graph_4 = {
    1: [6, 10],
    2: [7],
    3: [7],
    4: [8, 9],
    5: [6, 9, 14],
    6: [11, 13],
    7: [10, 12],
    8: [12],
    9: [13],
    10: [15],
    11: [15, 16],
    12: [16, 17],
    13: [17, 18],
    14: [18],
    15: [19],
    16: [19],
    17: [19],
    18: [19],
    19: [20],
    20: []
}

exec_times_4 = exec_times_3

run_example(4, graph_4, exec_times_4, T_s, T_c, T_r, P, P_s, K)

# EXAMPLE 5: 20 tasks with multiple entry and exit tasks
graph_5 = {
    1: [6, 10],
    2: [7],
    3: [7],
    4: [8, 9],
    5: [6, 9, 14],
    6: [11, 13],
    7: [10, 12],
    8: [12],
    9: [13],
    10: [15],
    11: [15, 16],
    12: [16],
    13: [16, 17],
    14: [17],
    15: [18],
    16: [18, 19, 20],
    17: [20],
    18: [],
    19: [],
    20: []
}

exec_times_5 = exec_times_3
run_example(5, graph_5, exec_times_5, T_s, T_c, T_r, P, P_s, K)


# In[ ]:




