import asyncio
import aristotlelib
import logging

async def prove_simple_theorem():
    # Set API key
    aristotlelib.set_api_key("arstl_Iv_EtlhgZBdYRexPasaAebRiELOFcqvy2sNOtjf5SlI")

    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    # Prove a simple theorem - ALL parameters must be keyword arguments
    project_id = await aristotlelib.Project.prove_from_file(
        input_file_path="examples/simple.lean", 
        output_file_path="examples/output.lean",
        wait_for_completion=True  # This will wait until the proof is complete
    )
    return project_id

async def get_project_status(project_id):
    # Load an existing project
    project = await aristotlelib.Project.from_id(project_id)

    # Check status
    print(f"Project status: {project.status}")

    if project.status == aristotlelib.ProjectStatus.COMPLETE:
        # if complete, download the solution
        await project.get_solution(output_path="examples/output.lean")
        return True
    else:
        return False

async def main():
    # Start the proof and get project ID
    # Since we set wait_for_completion=True, it will block until done
    project_id = await prove_simple_theorem()
    print(f"Project completed with ID: {project_id}")
    
    # Optionally check status (though it should be complete already)
    await get_project_status(project_id)

asyncio.run(main())