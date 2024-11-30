using DrWatson, Test

println("Starting tests")
ti = time()

include(srcdir("../headland_simulations/headland.jl"))

@testset "template tests" begin
    @test 1 == 1
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
