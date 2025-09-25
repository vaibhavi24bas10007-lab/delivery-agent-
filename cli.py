"""
Command-line interface for the autonomous delivery agent system.
"""

import argparse
import json
import time
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

from environment import (GridEnvironment, create_test_environment_small, 
                        create_test_environment_medium, create_test_environment_large,
                        create_test_environment_dynamic, Position)
from delivery_agent import DeliveryAgent, PlanningStrategy, DeliveryTask
from search_algorithms import compare_algorithms


def create_test_maps():
    """Create and save test maps."""
    maps = {
        'small': create_test_environment_small(),
        'medium': create_test_environment_medium(),
        'large': create_test_environment_large(),
        'dynamic': create_test_environment_dynamic()
    }
    
    for name, env in maps.items():
        filename = f"test_maps/{name}_map.json"
        env.save_to_file(filename)
        print(f"Created {name} map: {filename}")


def run_algorithm_comparison(env: GridEnvironment, start: Position, goal: Position,
                           start_time: int = 0) -> Dict[str, Any]:
    """Run comparison of all algorithms on a single problem."""
    print(f"Comparing algorithms from {start} to {goal}")
    print(f"Environment: {env.width}x{env.height} grid")
    
    # Validate positions
    if not env.is_valid_position(start.x, start.y):
        print(f"Error: Start position {start} is outside environment bounds")
        return {}
    if not env.is_valid_position(goal.x, goal.y):
        print(f"Error: Goal position {goal} is outside environment bounds")
        return {}
    
    results = compare_algorithms(env, start, goal, start_time)
    
    # Print summary table
    print("\nAlgorithm Comparison Results:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Success':<8} {'Path Length':<12} {'Cost':<10} {'Nodes':<8} {'Time (s)':<10}")
    print("-" * 80)
    
    successful_algorithms = []
    for name, result in results.items():
        success = "Yes" if result.success else "No"
        path_len = len(result.path) if result.success else "N/A"
        cost = f"{result.cost:.2f}" if result.success else "N/A"
        nodes = result.nodes_expanded
        time_taken = f"{result.time_taken:.4f}"
        
        if result.success:
            successful_algorithms.append((name, result.cost, result.time_taken))
        
        print(f"{name:<20} {success:<8} {path_len:<12} {cost:<10} {nodes:<8} {time_taken:<10}")
    
    # Show best performing algorithm
    if successful_algorithms:
        best_algorithm = min(successful_algorithms, key=lambda x: (x[1], x[2]))
        print(f"\nBest performing algorithm: {best_algorithm[0]} (cost: {best_algorithm[1]:.2f}, time: {best_algorithm[2]:.4f}s)")
    
    return results


def run_delivery_simulation(env: GridEnvironment, strategy: PlanningStrategy,
                          max_steps: int = 1000) -> Dict[str, Any]:
    """Run delivery simulation with specified strategy."""
    agent = DeliveryAgent(env, Position(0, 0), initial_fuel=100.0)
    agent.set_planning_strategy(strategy)
    
    # Add test tasks based on environment size
    tasks = []
    if env.width <= 10:  # Small map
        tasks = [
            DeliveryTask("package_1", Position(1, 1), Position(8, 8), priority=3),
            DeliveryTask("package_2", Position(2, 2), Position(7, 7), priority=2),
            DeliveryTask("package_3", Position(3, 3), Position(6, 6), priority=1),
        ]
    elif env.width <= 20:  # Medium map
        tasks = [
            DeliveryTask("package_1", Position(2, 2), Position(18, 18), priority=3),
            DeliveryTask("package_2", Position(4, 4), Position(16, 16), priority=2),
            DeliveryTask("package_3", Position(6, 6), Position(14, 14), priority=1),
        ]
    else:  # Large map
        tasks = [
            DeliveryTask("package_1", Position(5, 5), Position(45, 45), priority=3),
            DeliveryTask("package_2", Position(10, 10), Position(40, 40), priority=2),
            DeliveryTask("package_3", Position(15, 15), Position(35, 35), priority=1),
        ]
    
    for task in tasks:
        agent.add_delivery_task(task)
    
    print(f"Running delivery simulation with {strategy.value}")
    print(f"Environment: {env.width}x{env.height} grid")
    print(f"Tasks: {len(tasks)} packages to deliver")
    print(f"Max steps: {max_steps}")
    
    stats = agent.run_simulation(max_steps)
    
    print(f"\nSimulation completed:")
    print(f"  Deliveries completed: {stats['deliveries_completed']}/{len(tasks)}")
    print(f"  Steps taken: {stats['simulation_steps']}")
    print(f"  Distance traveled: {stats['total_distance_traveled']:.2f}")
    print(f"  Fuel consumed: {stats['total_fuel_consumed']:.2f}")
    print(f"  Planning time: {stats['total_planning_time']:.4f}s")
    print(f"  Replanning events: {stats['replanning_events']}")
    
    # Calculate efficiency metrics
    if stats['simulation_steps'] > 0:
        efficiency = stats['deliveries_completed'] / stats['simulation_steps'] * 100
        print(f"  Delivery efficiency: {efficiency:.2f}% (deliveries per step)")
    
    return stats


def run_comprehensive_experiment():
    """Run comprehensive experiment comparing all algorithms on all maps."""
    print("Running comprehensive experiment...")
    
    # Load test maps
    maps = {}
    for name in ['small', 'medium', 'large', 'dynamic']:
        try:
            maps[name] = GridEnvironment.load_from_file(f"test_maps/{name}_map.json")
            print(f"Loaded {name} map")
        except FileNotFoundError:
            print(f"Creating {name} map...")
            if name == 'small':
                maps[name] = create_test_environment_small()
            elif name == 'medium':
                maps[name] = create_test_environment_medium()
            elif name == 'large':
                maps[name] = create_test_environment_large()
            elif name == 'dynamic':
                maps[name] = create_test_environment_dynamic()
            maps[name].save_to_file(f"test_maps/{name}_map.json")
    
    # Define test scenarios
    scenarios = {
        'small': [(Position(0, 0), Position(9, 9))],
        'medium': [(Position(0, 0), Position(19, 19)), (Position(5, 5), Position(15, 15))],
        'large': [(Position(0, 0), Position(49, 49)), (Position(10, 10), Position(40, 40))],
        'dynamic': [(Position(0, 0), Position(14, 14)), (Position(1, 1), Position(13, 13))]
    }
    
    # Run experiments
    all_results = {}
    
    for map_name, env in maps.items():
        print(f"\nTesting on {map_name} map...")
        map_results = {}
        
        for i, (start, goal) in enumerate(scenarios[map_name]):
            print(f"  Scenario {i+1}: {start} -> {goal}")
            results = run_algorithm_comparison(env, start, goal)
            map_results[f"scenario_{i+1}"] = results
        
        all_results[map_name] = map_results
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for map_name, map_data in all_results.items():
            json_results[map_name] = {}
            for scenario_name, scenario_data in map_data.items():
                json_results[map_name][scenario_name] = {}
                for alg_name, result in scenario_data.items():
                    json_results[map_name][scenario_name][alg_name] = {
                        'success': result.success,
                        'path_length': len(result.path) if result.success else 0,
                        'cost': result.cost,
                        'nodes_expanded': result.nodes_expanded,
                        'time_taken': result.time_taken
                    }
        
        json.dump(json_results, f, indent=2)
    
    print("\nExperiment completed. Results saved to experiment_results.json")
    return all_results


def generate_analysis_plots(results: Dict[str, Any]):
    """Generate analysis plots from experiment results."""
    print("Generating analysis plots...")
    
    # Extract data for plotting
    algorithms = ['BFS', 'A* Manhattan', 'Hill Climbing', 'Simulated Annealing']  # FIXED
    map_sizes = ['small', 'medium', 'large', 'dynamic']
    
    # Plot 1: Success rate by algorithm
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Algorithm Performance Analysis')
    
    for i, map_name in enumerate(map_sizes):
        ax = axes[i//2, i%2]
        success_rates = []
        
        for alg in algorithms:
            total_scenarios = 0
            successful_scenarios = 0
            
            for scenario_name, scenario_data in results[map_name].items():
                if alg in scenario_data:
                    total_scenarios += 1
                    if scenario_data[alg]['success']:
                        successful_scenarios += 1
            
            success_rate = (successful_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
            success_rates.append(success_rate)
        
        ax.bar(algorithms, success_rates)
        ax.set_title(f'{map_name.capitalize()} Map')
        ax.set_ylabel('Success Rate (%)')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('algorithm_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Average path cost by algorithm
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for map_name in map_sizes:
        avg_costs = []
        for alg in algorithms:
            costs = []
            for scenario_name, scenario_data in results[map_name].items():
                if alg in scenario_data and scenario_data[alg]['success']:
                    costs.append(scenario_data[alg]['cost'])
            
            avg_cost = np.mean(costs) if costs else 0
            avg_costs.append(avg_cost)
        
        ax.plot(algorithms, avg_costs, marker='o', label=map_name.capitalize())
    
    ax.set_title('Average Path Cost by Algorithm')
    ax.set_ylabel('Average Cost')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_costs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots saved as algorithm_success_rates.png and algorithm_costs.png")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Autonomous Delivery Agent System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py create-maps
  python cli.py compare --map small --start 0,0 --goal 9,9
  python cli.py simulate --map medium --strategy "A* Manhattan" --max-steps 500
  python cli.py experiment
  python cli.py analyze
        """
    )
    parser.add_argument('command', choices=['create-maps', 'compare', 'simulate', 'experiment', 'analyze'],
                       help='Command to execute')
    parser.add_argument('--map', choices=['small', 'medium', 'large', 'dynamic'],
                       help='Map to use for comparison/simulation')
    parser.add_argument('--strategy', choices=[s.value for s in PlanningStrategy],
                       help='Planning strategy for simulation')
    parser.add_argument('--start', type=str, help='Start position (format: x,y)')
    parser.add_argument('--goal', type=str, help='Goal position (format: x,y)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum simulation steps')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'create-maps':
            import os
            os.makedirs('test_maps', exist_ok=True)
            create_test_maps()
        
        elif args.command == 'compare':
            if not args.map or not args.start or not args.goal:
                print("Error: --map, --start, and --goal are required for comparison")
                return
            
            # Parse positions with validation
            try:
                start_coords = tuple(map(int, args.start.split(',')))
                goal_coords = tuple(map(int, args.goal.split(',')))
                if len(start_coords) != 2 or len(goal_coords) != 2:
                    raise ValueError("Invalid coordinate format")
            except ValueError as e:
                print(f"Error: Invalid position format. Use x,y (e.g., 0,0)")
                return
            
            start_pos = Position(start_coords[0], start_coords[1])
            goal_pos = Position(goal_coords[0], goal_coords[1])
            
            # Load map with error handling
            try:
                env = GridEnvironment.load_from_file(f"test_maps/{args.map}_map.json")
            except FileNotFoundError:
                print(f"Error: Map file 'test_maps/{args.map}_map.json' not found.")
                print("Run 'python cli.py create-maps' first to create test maps.")
                return
            
            run_algorithm_comparison(env, start_pos, goal_pos)
        
        elif args.command == 'simulate':
            if not args.map or not args.strategy:
                print("Error: --map and --strategy are required for simulation")
                return
            
            # Load map with error handling
            try:
                env = GridEnvironment.load_from_file(f"test_maps/{args.map}_map.json")
            except FileNotFoundError:
                print(f"Error: Map file 'test_maps/{args.map}_map.json' not found.")
                print("Run 'python cli.py create-maps' first to create test maps.")
                return
            
            strategy = PlanningStrategy(args.strategy)
            run_delivery_simulation(env, strategy, args.max_steps)
        
        elif args.command == 'experiment':
            print("Starting comprehensive experiment...")
            run_comprehensive_experiment()
        
        elif args.command == 'analyze':
            try:
                with open('experiment_results.json', 'r') as f:
                    results = json.load(f)
                generate_analysis_plots(results)
            except FileNotFoundError:
                print("Error: experiment_results.json not found. Run 'experiment' command first.")
            except json.JSONDecodeError:
                print("Error: Invalid JSON in experiment_results.json")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
