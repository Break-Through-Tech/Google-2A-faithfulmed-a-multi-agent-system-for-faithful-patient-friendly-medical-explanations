import asyncio

MAX_ITERATIONS = 2


async def refinement_loop(verify, refine):
    feedback = await verify()
    iterations = 0

    while not feedback["overall_pass"] and iterations < MAX_ITERATIONS:
        iterations += 1

        await refine()

        feedback = await verify()

    return {
        "overall_pass": feedback["overall_pass"],
        "iterations": iterations,
    }


async def main():
    # Test 1: Passes after the first refinement
    verification_results = iter([False, True])
    refine_calls = 0

    async def verify_first():
        return {"overall_pass": next(verification_results)}

    async def refine_first():
        nonlocal refine_calls
        refine_calls += 1

    result = await refinement_loop(verify_first, refine_first)

    assert result["overall_pass"] is True
    assert result["iterations"] == 1
    assert refine_calls == 1

    print("Test 1 passed: stopped after one successful refinement.")

    # Test 2: Never passes, must stop after two refinements
    verification_results = iter([False, False, False])
    refine_calls = 0

    async def verify_second():
        return {"overall_pass": next(verification_results)}

    async def refine_second():
        nonlocal refine_calls
        refine_calls += 1

    result = await refinement_loop(verify_second, refine_second)

    assert result["overall_pass"] is False
    assert result["iterations"] == 2
    assert refine_calls == 2

    print("Test 2 passed: stopped after maximum two refinements.")

    # Test 3: Passes immediately, no refinement needed
    verification_results = iter([True])
    refine_calls = 0

    async def verify_third():
        return {"overall_pass": next(verification_results)}

    async def refine_third():
        nonlocal refine_calls
        refine_calls += 1

    result = await refinement_loop(verify_third, refine_third)

    assert result["overall_pass"] is True
    assert result["iterations"] == 0
    assert refine_calls == 0

    print("Test 3 passed: skipped refinement when already verified.")


if __name__ == "__main__":
    asyncio.run(main())