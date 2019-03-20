
#pragma once


#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/dot.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/softmax.hpp>
#include "test_tools.hpp"
#include "random.hpp"

void run_ngraph()
{

    ngraph::Shape data_batch_shape{2, 1, 28, 28};
    ngraph::Shape flat_1_shape{2, 1*28*28};
    ngraph::Shape fc_1_shape{1*28*28, 128};
    ngraph::Shape fc_2_shape{128, 64};
    ngraph::Shape fc_3_shape{64, 10};

    auto data_batch = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, data_batch_shape);
    auto fc_1_weight = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, fc_1_shape);
    auto fc_2_weight = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, fc_2_shape);
    auto fc_3_weight = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, fc_3_shape);

    auto flat_1 = std::make_shared<ngraph::op::Reshape>(data_batch, ngraph::AxisVector{0,1,2,3}, flat_1_shape);
    auto fc_1 = std::make_shared<ngraph::op::Dot>(flat_1, fc_1_weight);
    auto act_1 = std::make_shared<ngraph::op::Relu>(fc_1);
    auto fc_2 = std::make_shared<ngraph::op::Dot>(act_1, fc_2_weight);
    auto act_2 = std::make_shared<ngraph::op::Relu>(fc_2);
    auto fc_3 = std::make_shared<ngraph::op::Dot>(act_2, fc_3_weight);
    auto softmax_1 = std::make_shared<ngraph::op::Softmax>(fc_3, ngraph::AxisSet{1});

    auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_1},
                                   ngraph::ParameterVector{data_batch, fc_1_weight,
                                                   fc_2_weight, fc_3_weight});

    ngraph::test::Uniform<float> rng(-0.5f, 0.5f);
    std::vector<std::vector<float>> fprop_args;
    for (std::shared_ptr<ngraph::op::Parameter> param : f->get_parameters())
    {
        std::vector<float> tensor_val(ngraph::shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        fprop_args.push_back(tensor_val);
    }
    auto fprop_results = execute(f, fprop_args, "INTERPRETER");
    std::cout << "fprop results:" << std::endl;
    for (auto& result : fprop_results) {
        std::cout << ngraph::vector_to_string(result) << std::endl;
    }
}
