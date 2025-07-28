#!/usr/bin/env python3
"""
Load Testing Script for Iris Classification API
Tests concurrent inference capabilities and identifies bottlenecks.
"""

import asyncio
import aiohttp
import time
import json
import random
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_random_features(self) -> Dict[str, float]:
        """Generate random iris features for testing"""
        return {
            "sepal_length": round(random.uniform(4.0, 8.0), 2),
            "sepal_width": round(random.uniform(2.0, 5.0), 2),
            "petal_length": round(random.uniform(1.0, 7.0), 2),
            "petal_width": round(random.uniform(0.1, 3.0), 2)
        }
    
    async def single_prediction_request(self, request_id: int) -> Dict[str, Any]:
        """Make a single prediction request and measure performance"""
        start_time = time.time()
        
        try:
            features = self.generate_random_features()
            
            async with self.session.post(
                f"{self.base_url}/predict",
                json=features,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "request_id": request_id,
                        "success": True,
                        "response_time": response_time,
                        "status_code": response.status,
                        "prediction": result.get("prediction"),
                        "confidence": result.get("confidence")
                    }
                else:
                    error_text = await response.text()
                    return {
                        "request_id": request_id,
                        "success": False,
                        "response_time": response_time,
                        "status_code": response.status,
                        "error": error_text
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "request_id": request_id,
                "success": False,
                "response_time": response_time,
                "status_code": 0,
                "error": str(e)
            }
    
    async def batch_prediction_request(self, batch_size: int) -> Dict[str, Any]:
        """Make a batch prediction request"""
        start_time = time.time()
        
        try:
            features_list = [self.generate_random_features() for _ in range(batch_size)]
            
            async with self.session.post(
                f"{self.base_url}/predict/batch",
                json=features_list,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    results = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "batch_size": batch_size,
                        "predictions_count": len(results),
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "batch_size": batch_size,
                        "status_code": response.status,
                        "error": error_text
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "batch_size": batch_size,
                "status_code": 0,
                "error": str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the API"""
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get metrics: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
    
    async def concurrent_load_test(self, num_requests: int, max_concurrent: int) -> Dict[str, Any]:
        """Run concurrent load test with limited concurrency"""
        logger.info(f"Starting concurrent load test: {num_requests} requests, {max_concurrent} max concurrent")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(request_id: int):
            async with semaphore:
                return await self.single_prediction_request(request_id)
        
        start_time = time.time()
        
        # Create tasks for all requests
        tasks = [limited_request(i) for i in range(num_requests)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_requests = [r for r in results if not (isinstance(r, dict) and r.get("success", False))]
        
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = sorted(response_times)[int(0.95 * len(response_times))] if response_times else 0
            p99_response_time = sorted(response_times)[int(0.99 * len(response_times))] if response_times else 0
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
        
        return {
            "test_type": "concurrent_single_requests",
            "total_requests": num_requests,
            "max_concurrent": max_concurrent,
            "total_time": round(total_time, 4),
            "requests_per_second": round(num_requests / total_time, 2),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": round(len(successful_requests) / num_requests * 100, 2),
            "avg_response_time": round(avg_response_time, 4),
            "median_response_time": round(median_response_time, 4),
            "p95_response_time": round(p95_response_time, 4),
            "p99_response_time": round(p99_response_time, 4)
        }
    
    async def batch_load_test(self, num_batches: int, batch_size: int) -> Dict[str, Any]:
        """Run batch prediction load test"""
        logger.info(f"Starting batch load test: {num_batches} batches, {batch_size} items per batch")
        
        start_time = time.time()
        
        # Create tasks for all batch requests
        tasks = [self.batch_prediction_request(batch_size) for _ in range(num_batches)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_requests = [r for r in results if not (isinstance(r, dict) and r.get("success", False))]
        
        total_predictions = sum(r.get("predictions_count", 0) for r in successful_requests)
        
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
        else:
            avg_response_time = 0
        
        return {
            "test_type": "batch_requests",
            "total_batches": num_batches,
            "batch_size": batch_size,
            "total_predictions": total_predictions,
            "total_time": round(total_time, 4),
            "batches_per_second": round(num_batches / total_time, 2),
            "predictions_per_second": round(total_predictions / total_time, 2),
            "successful_batches": len(successful_requests),
            "failed_batches": len(failed_requests),
            "success_rate": round(len(successful_requests) / num_batches * 100, 2),
            "avg_batch_response_time": round(avg_response_time, 4)
        }
    
    async def run_comprehensive_test(self):
        """Run a comprehensive load test with different scenarios"""
        logger.info("Starting comprehensive load test")
        
        # Get initial metrics
        initial_metrics = await self.get_metrics()
        logger.info(f"Initial metrics: {initial_metrics}")
        
        test_results = []
        
        # Test 1: Low concurrency baseline
        result1 = await self.concurrent_load_test(50, 5)
        test_results.append(result1)
        logger.info(f"Test 1 (Low concurrency): {result1}")
        
        # Test 2: Medium concurrency
        result2 = await self.concurrent_load_test(100, 20)
        test_results.append(result2)
        logger.info(f"Test 2 (Medium concurrency): {result2}")
        
        # Test 3: High concurrency
        result3 = await self.concurrent_load_test(200, 50)
        test_results.append(result3)
        logger.info(f"Test 3 (High concurrency): {result3}")
        
        # Test 4: Very high concurrency (stress test)
        result4 = await self.concurrent_load_test(500, 100)
        test_results.append(result4)
        logger.info(f"Test 4 (Stress test): {result4}")
        
        # Test 5: Batch processing test
        result5 = await self.batch_load_test(20, 10)
        test_results.append(result5)
        logger.info(f"Test 5 (Batch processing): {result5}")
        
        # Test 6: Large batch processing
        result6 = await self.batch_load_test(10, 50)
        test_results.append(result6)
        logger.info(f"Test 6 (Large batch): {result6}")
        
        # Get final metrics
        final_metrics = await self.get_metrics()
        logger.info(f"Final metrics: {final_metrics}")
        
        return {
            "initial_metrics": initial_metrics,
            "test_results": test_results,
            "final_metrics": final_metrics,
            "summary": self._generate_summary(test_results, initial_metrics, final_metrics)
        }
    
    def _generate_summary(self, test_results: List[Dict], initial_metrics: Dict, final_metrics: Dict) -> Dict[str, Any]:
        """Generate a summary of test results and identify bottlenecks"""
        concurrent_tests = [r for r in test_results if r["test_type"] == "concurrent_single_requests"]
        batch_tests = [r for r in test_results if r["test_type"] == "batch_requests"]
        
        # Analyze performance degradation
        if len(concurrent_tests) >= 2:
            baseline = concurrent_tests[0]
            stress_test = concurrent_tests[-1]
            
            throughput_degradation = (
                (baseline["requests_per_second"] - stress_test["requests_per_second"]) 
                / baseline["requests_per_second"] * 100
            )
            
            latency_increase = (
                (stress_test["avg_response_time"] - baseline["avg_response_time"]) 
                / baseline["avg_response_time"] * 100
            )
        else:
            throughput_degradation = latency_increase = 0
        
        # Identify bottlenecks
        bottlenecks = []
        
        if any(r["success_rate"] < 95 for r in concurrent_tests):
            bottlenecks.append("High error rate under load - possible resource exhaustion")
        
        if throughput_degradation > 50:
            bottlenecks.append("Significant throughput degradation under high concurrency")
        
        if latency_increase > 200:
            bottlenecks.append("High latency increase under load - possible GIL contention or resource limits")
        
        max_concurrent_test = max(concurrent_tests, key=lambda x: x["max_concurrent"])
        if max_concurrent_test["p99_response_time"] > 5.0:
            bottlenecks.append("High P99 latency - possible resource contention")
        
        return {
            "max_throughput": max(r["requests_per_second"] for r in concurrent_tests),
            "best_batch_throughput": max(r["predictions_per_second"] for r in batch_tests) if batch_tests else 0,
            "throughput_degradation_percent": round(throughput_degradation, 2),
            "latency_increase_percent": round(latency_increase, 2),
            "identified_bottlenecks": bottlenecks,
            "recommendations": self._generate_recommendations(bottlenecks, final_metrics)
        }
    
    def _generate_recommendations(self, bottlenecks: List[str], metrics: Dict) -> List[str]:
        """Generate recommendations based on identified bottlenecks"""
        recommendations = []
        
        if "High error rate under load" in str(bottlenecks):
            recommendations.append("Increase resource limits (CPU/Memory) in Kubernetes deployment")
            recommendations.append("Implement circuit breaker pattern for graceful degradation")
        
        if "throughput degradation" in str(bottlenecks):
            recommendations.append("Increase number of worker processes/threads")
            recommendations.append("Consider horizontal pod autoscaling")
        
        if "High latency increase" in str(bottlenecks):
            recommendations.append("Optimize model inference code")
            recommendations.append("Implement model caching strategies")
            recommendations.append("Consider using async/await throughout the pipeline")
        
        if "P99 latency" in str(bottlenecks):
            recommendations.append("Implement request queuing with priority handling")
            recommendations.append("Add connection pooling for database/external services")
        
        cache_efficiency = metrics.get("cache_hit_ratio", 0)
        if cache_efficiency < 0.5:
            recommendations.append("Increase cache size for better hit ratio")
        
        return recommendations

async def main():
    parser = argparse.ArgumentParser(description="Load test the Iris Classification API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests for simple test")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    async with LoadTester(args.url) as tester:
        if args.comprehensive:
            results = await tester.run_comprehensive_test()
            print("\n" + "="*80)
            print("COMPREHENSIVE LOAD TEST RESULTS")
            print("="*80)
            print(json.dumps(results, indent=2))
        else:
            result = await tester.concurrent_load_test(args.requests, args.concurrency)
            print("\n" + "="*50)
            print("LOAD TEST RESULTS")
            print("="*50)
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())