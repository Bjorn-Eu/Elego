cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Selfplay)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Torch REQUIRED)

add_executable(Selfplay
    main.cc
    gameplay/gamestate.cc
    gameplay/board.cc
    gameplay/move.cc
    gameplay/group.cc
    gameplay/gamestate.h
    gameplay/board.h
    gameplay/move.h
    gameplay/group.h
    engine.cc
    engine.h
    encoders/extended_encoder.cc
    agents/policy_agent.cc
    agents/policy_agent.h
    agents/random_agent.cc
    agents/random_agent.h
    agents/node.cc
    agents/node.h
    agents/mcts.cc
    agents/mcts.h
    agents/node.h
    agents/branch.h
    agents/test_nets.h
    agents/test_policy.h
    network_evaluation.cc
    network_evaluation.h
)

target_link_libraries(Selfplay "${TORCH_LIBRARIES}")
set_property(TARGET  PROPERTY CXX_STANDARD 17)
