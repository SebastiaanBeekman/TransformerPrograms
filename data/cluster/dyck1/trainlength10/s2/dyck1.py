import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck1/trainlength10/s2/dyck1_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 7

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"(", ")"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 6

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 17
        elif token in {"<s>"}:
            return position == 12

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"("}:
            return k_token == "("
        elif q_token in {")"}:
            return k_token == ")"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(token, position):
        if token in {"<s>", "(", ")"}:
            return position == 1

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4, 10, 11, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {7}:
            return k_position == 17
        elif q_position in {8}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 4
        elif q_position in {12}:
            return k_position == 10

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 2, 11, 12, 14, 15, 16}:
            return k_position == 2
        elif q_position in {1, 3, 5, 7, 8, 9}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {17, 6}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 18

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 1, 2, 17}:
            return k_position == 1
        elif q_position in {3, 5, 8, 9, 10, 13, 14, 19}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 13
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {16, 15}:
            return k_position == 0
        elif q_position in {18}:
            return k_position == 11

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 5
        elif token in {")"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 19

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"(", ")"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == "("

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 11
        elif q_position in {1, 6, 10, 13, 14, 16, 17, 18}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {8, 3, 4, 5}:
            return k_position == 6
        elif q_position in {9, 19}:
            return k_position == 17
        elif q_position in {11, 12}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 7

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 9
        elif token in {")"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_token, k_token):
        if q_token in {"<s>", "("}:
            return k_token == ""
        elif q_token in {")"}:
            return k_token == "("

    num_attn_0_4_pattern = select(tokens, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {9, 3}:
            return k_position == 1
        elif q_position in {17, 4}:
            return k_position == 16
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {19, 6}:
            return k_position == 12
        elif q_position in {13, 7}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {10, 11}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {14, 15}:
            return k_position == 10
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 9

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 16
        elif q_position in {1, 3, 5, 10, 13, 16, 17, 19}:
            return k_position == 2
        elif q_position in {2, 6}:
            return k_position == 5
        elif q_position in {11, 4}:
            return k_position == 19
        elif q_position in {18, 12, 14, 7}:
            return k_position == 0
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 4
        elif q_position in {15}:
            return k_position == 3

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_position, k_position):
        if q_position in {0, 16}:
            return k_position == 15
        elif q_position in {1, 3, 5}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {19, 11, 6, 14}:
            return k_position == 13
        elif q_position in {9, 7}:
            return k_position == 1
        elif q_position in {8}:
            return k_position == 19
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 5
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 16

    num_attn_0_7_pattern = select(positions, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_3_output):
        key = (attn_0_0_output, attn_0_3_output)
        if key in {
            ("(", "("),
            ("(", ")"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            ("<s>", "("),
            ("<s>", ")"),
        }:
            return 16
        return 9

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        return 3

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_4_output):
        key = (num_attn_0_1_output, num_attn_0_4_output)
        return 11

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_6_output, num_attn_0_0_output):
        key = (num_attn_0_6_output, num_attn_0_0_output)
        return 11

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_7_output, token):
        if attn_0_7_output in {"("}:
            return token == "("
        elif attn_0_7_output in {")"}:
            return token == ""
        elif attn_0_7_output in {"<s>"}:
            return token == ")"

    attn_1_0_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, attn_0_4_output):
        if attn_0_3_output in {"("}:
            return attn_0_4_output == ""
        elif attn_0_3_output in {"<s>", ")"}:
            return attn_0_4_output == "("

    attn_1_1_pattern = select_closest(attn_0_4_outputs, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"<s>", "("}:
            return attn_0_0_output == ")"
        elif attn_0_3_output in {")"}:
            return attn_0_0_output == "("

    attn_1_2_pattern = select_closest(attn_0_0_outputs, attn_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, attn_0_4_output):
        if attn_0_2_output in {"<s>", "("}:
            return attn_0_4_output == ""
        elif attn_0_2_output in {")"}:
            return attn_0_4_output == ")"

    attn_1_3_pattern = select_closest(attn_0_4_outputs, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 1, 2, 4, 11, 12, 14, 16, 17}:
            return k_position == 1
        elif q_position in {3, 15}:
            return k_position == 14
        elif q_position in {5}:
            return k_position == 11
        elif q_position in {8, 10, 6}:
            return k_position == 2
        elif q_position in {13, 7}:
            return k_position == 19
        elif q_position in {9}:
            return k_position == 17
        elif q_position in {18}:
            return k_position == 4
        elif q_position in {19}:
            return k_position == 15

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 18
        elif attn_0_0_output in {")"}:
            return position == 16
        elif attn_0_0_output in {"<s>"}:
            return position == 2

    attn_1_5_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, attn_0_0_output):
        if position in {0, 1, 2, 7, 11, 13}:
            return attn_0_0_output == ")"
        elif position in {17, 3}:
            return attn_0_0_output == "<pad>"
        elif position in {4, 6}:
            return attn_0_0_output == "<s>"
        elif position in {9, 5}:
            return attn_0_0_output == "("
        elif position in {8, 10, 12, 14, 15, 16, 18, 19}:
            return attn_0_0_output == ""

    attn_1_6_pattern = select_closest(attn_0_0_outputs, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_2_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_7_output, token):
        if attn_0_7_output in {"<s>", "("}:
            return token == "("
        elif attn_0_7_output in {")"}:
            return token == ""

    attn_1_7_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, num_mlp_0_1_output):
        if position in {0, 11, 15}:
            return num_mlp_0_1_output == 15
        elif position in {1, 19}:
            return num_mlp_0_1_output == 6
        elif position in {2}:
            return num_mlp_0_1_output == 8
        elif position in {16, 3}:
            return num_mlp_0_1_output == 17
        elif position in {4}:
            return num_mlp_0_1_output == 13
        elif position in {5}:
            return num_mlp_0_1_output == 1
        elif position in {18, 6}:
            return num_mlp_0_1_output == 11
        elif position in {8, 7}:
            return num_mlp_0_1_output == 19
        elif position in {9}:
            return num_mlp_0_1_output == 5
        elif position in {10}:
            return num_mlp_0_1_output == 18
        elif position in {12, 14}:
            return num_mlp_0_1_output == 14
        elif position in {13}:
            return num_mlp_0_1_output == 4
        elif position in {17}:
            return num_mlp_0_1_output == 9

    num_attn_1_0_pattern = select(num_mlp_0_1_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_1_output, token):
        if num_mlp_0_1_output in {0, 1, 2, 7, 9}:
            return token == ""
        elif num_mlp_0_1_output in {3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19}:
            return token == ")"
        elif num_mlp_0_1_output in {14}:
            return token == "<s>"

    num_attn_1_1_pattern = select(tokens, num_mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_6_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(token, mlp_0_1_output):
        if token in {"("}:
            return mlp_0_1_output == 9
        elif token in {")"}:
            return mlp_0_1_output == 17
        elif token in {"<s>"}:
            return mlp_0_1_output == 19

    num_attn_1_2_pattern = select(mlp_0_1_outputs, tokens, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_2_output, num_mlp_0_1_output):
        if attn_0_2_output in {"("}:
            return num_mlp_0_1_output == 0
        elif attn_0_2_output in {")"}:
            return num_mlp_0_1_output == 4
        elif attn_0_2_output in {"<s>"}:
            return num_mlp_0_1_output == 10

    num_attn_1_3_pattern = select(
        num_mlp_0_1_outputs, attn_0_2_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"(", ")"}:
            return attn_0_0_output == ")"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_0_output == ""

    num_attn_1_4_pattern = select(attn_0_0_outputs, attn_0_3_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_5_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_0_output, attn_0_4_output):
        if attn_0_0_output in {"<s>", "(", ")"}:
            return attn_0_4_output == ""

    num_attn_1_5_pattern = select(attn_0_4_outputs, attn_0_0_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_3_output, attn_0_4_output):
        if attn_0_3_output in {"("}:
            return attn_0_4_output == ""
        elif attn_0_3_output in {")"}:
            return attn_0_4_output == "<pad>"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_4_output == ")"

    num_attn_1_6_pattern = select(attn_0_4_outputs, attn_0_3_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_5_output, token):
        if attn_0_5_output in {"<s>", "(", ")"}:
            return token == ""

    num_attn_1_7_pattern = select(tokens, attn_0_5_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_0_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_4_output, attn_1_7_output):
        key = (attn_1_4_output, attn_1_7_output)
        if key in {("(", "(")}:
            return 1
        elif key in {("<s>", "<s>")}:
            return 15
        return 3

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_4_outputs, attn_1_7_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_6_output):
        key = attn_1_6_output
        if key in {""}:
            return 1
        return 11

    mlp_1_1_outputs = [mlp_1_1(k0) for k0 in attn_1_6_outputs]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_2_output):
        key = (num_attn_1_7_output, num_attn_1_2_output)
        if key in {
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
            (9, 39),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 33),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (12, 29),
            (12, 30),
            (12, 31),
            (12, 32),
            (12, 33),
            (12, 34),
            (12, 35),
            (12, 36),
            (12, 37),
            (12, 38),
            (12, 39),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 36),
            (14, 37),
            (14, 38),
            (14, 39),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (16, 37),
            (16, 38),
            (16, 39),
            (17, 39),
        }:
            return 17
        elif key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (7, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 23),
            (11, 24),
            (11, 25),
            (11, 26),
            (12, 21),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (14, 24),
            (14, 25),
            (14, 26),
            (14, 27),
            (14, 28),
            (14, 29),
            (14, 30),
            (14, 31),
            (14, 32),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (16, 28),
            (16, 29),
            (16, 30),
            (16, 31),
            (16, 32),
            (16, 33),
            (16, 34),
            (16, 35),
            (16, 36),
            (17, 30),
            (17, 31),
            (17, 32),
            (17, 33),
            (17, 34),
            (17, 35),
            (17, 36),
            (17, 37),
            (17, 38),
            (18, 31),
            (18, 32),
            (18, 33),
            (18, 34),
            (18, 35),
            (18, 36),
            (18, 37),
            (18, 38),
            (18, 39),
            (19, 33),
            (19, 34),
            (19, 35),
            (19, 36),
            (19, 37),
            (19, 38),
            (19, 39),
            (20, 35),
            (20, 36),
            (20, 37),
            (20, 38),
            (20, 39),
            (21, 37),
            (21, 38),
            (21, 39),
            (22, 39),
        }:
            return 18
        elif key in {
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 15),
        }:
            return 19
        return 14

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_4_output, num_attn_1_5_output):
        key = (num_attn_1_4_output, num_attn_1_5_output)
        if key in {
            (1, 2),
            (2, 2),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (5, 2),
            (5, 3),
            (5, 4),
            (6, 2),
            (6, 3),
            (6, 4),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (20, 10),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (24, 11),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (25, 12),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (26, 11),
            (26, 12),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (27, 13),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (28, 13),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 13),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (30, 13),
            (30, 14),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
            (31, 13),
            (31, 14),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (32, 8),
            (32, 9),
            (32, 10),
            (32, 11),
            (32, 12),
            (32, 13),
            (32, 14),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 8),
            (33, 9),
            (33, 10),
            (33, 11),
            (33, 12),
            (33, 13),
            (33, 14),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 9),
            (35, 10),
            (35, 11),
            (35, 12),
            (35, 13),
            (35, 14),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (36, 9),
            (36, 10),
            (36, 11),
            (36, 12),
            (36, 13),
            (36, 14),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (37, 9),
            (37, 10),
            (37, 11),
            (37, 12),
            (37, 13),
            (37, 14),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (38, 9),
            (38, 10),
            (38, 11),
            (38, 12),
            (38, 13),
            (38, 14),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 10),
            (39, 11),
            (39, 12),
            (39, 13),
            (39, 14),
        }:
            return 15
        elif key in {
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (15, 0),
            (16, 0),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
            (20, 0),
            (20, 1),
            (21, 0),
            (21, 1),
            (22, 0),
            (22, 1),
            (23, 0),
            (23, 1),
            (24, 0),
            (24, 1),
            (25, 0),
            (25, 1),
            (26, 0),
            (26, 1),
            (27, 0),
            (27, 1),
            (28, 0),
            (28, 1),
            (29, 0),
            (29, 1),
            (30, 0),
            (30, 1),
            (31, 0),
            (31, 1),
            (32, 0),
            (32, 1),
            (33, 0),
            (33, 1),
            (34, 0),
            (34, 1),
            (35, 0),
            (35, 1),
            (36, 0),
            (36, 1),
            (37, 0),
            (37, 1),
            (38, 0),
            (38, 1),
            (39, 0),
            (39, 1),
        }:
            return 16
        elif key in {
            (0, 1),
            (0, 2),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (9, 1),
            (10, 1),
            (11, 1),
            (12, 1),
            (12, 2),
            (13, 1),
            (13, 2),
            (14, 1),
            (14, 2),
            (15, 1),
            (15, 2),
            (16, 1),
            (16, 2),
            (17, 2),
            (18, 2),
            (19, 2),
            (20, 2),
            (21, 2),
            (22, 2),
            (23, 2),
            (24, 2),
            (25, 2),
            (26, 2),
            (27, 2),
            (28, 2),
            (29, 2),
            (30, 2),
            (31, 2),
            (32, 2),
            (33, 2),
            (34, 2),
            (35, 2),
            (36, 2),
            (36, 3),
            (37, 2),
            (37, 3),
            (38, 2),
            (38, 3),
            (39, 2),
            (39, 3),
        }:
            return 11
        elif key in {
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 4),
            (4, 5),
        }:
            return 7
        return 3

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_5_output, num_mlp_1_0_output):
        if attn_1_5_output in {"(", ")"}:
            return num_mlp_1_0_output == 13
        elif attn_1_5_output in {"<s>"}:
            return num_mlp_1_0_output == 17

    attn_2_0_pattern = select_closest(
        num_mlp_1_0_outputs, attn_1_5_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_7_output, attn_0_5_output):
        if attn_0_7_output in {"<s>", "("}:
            return attn_0_5_output == ""
        elif attn_0_7_output in {")"}:
            return attn_0_5_output == ")"

    attn_2_1_pattern = select_closest(attn_0_5_outputs, attn_0_7_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_7_output, attn_1_6_output):
        if attn_1_7_output in {"("}:
            return attn_1_6_output == "("
        elif attn_1_7_output in {")"}:
            return attn_1_6_output == "<s>"
        elif attn_1_7_output in {"<s>"}:
            return attn_1_6_output == ""

    attn_2_2_pattern = select_closest(attn_1_6_outputs, attn_1_7_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(num_mlp_1_1_output, attn_0_7_output):
        if num_mlp_1_1_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            13,
            14,
            15,
            16,
            17,
            19,
        }:
            return attn_0_7_output == "("
        elif num_mlp_1_1_output in {11}:
            return attn_0_7_output == ""
        elif num_mlp_1_1_output in {18, 12}:
            return attn_0_7_output == ")"

    attn_2_3_pattern = select_closest(
        attn_0_7_outputs, num_mlp_1_1_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_1_1_output, mlp_1_0_output):
        if mlp_1_1_output in {0, 1, 2, 6}:
            return mlp_1_0_output == 0
        elif mlp_1_1_output in {3}:
            return mlp_1_0_output == 1
        elif mlp_1_1_output in {4}:
            return mlp_1_0_output == 4
        elif mlp_1_1_output in {19, 5}:
            return mlp_1_0_output == 17
        elif mlp_1_1_output in {7, 8, 10, 11, 16, 18}:
            return mlp_1_0_output == 3
        elif mlp_1_1_output in {9}:
            return mlp_1_0_output == 9
        elif mlp_1_1_output in {12, 13}:
            return mlp_1_0_output == 10
        elif mlp_1_1_output in {14}:
            return mlp_1_0_output == 8
        elif mlp_1_1_output in {17, 15}:
            return mlp_1_0_output == 13

    attn_2_4_pattern = select_closest(mlp_1_0_outputs, mlp_1_1_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, mlp_0_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"<s>", "(", ")"}:
            return attn_0_0_output == "("

    attn_2_5_pattern = select_closest(attn_0_0_outputs, attn_0_3_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(mlp_1_1_output, num_mlp_1_1_output):
        if mlp_1_1_output in {0}:
            return num_mlp_1_1_output == 7
        elif mlp_1_1_output in {1, 18}:
            return num_mlp_1_1_output == 1
        elif mlp_1_1_output in {2, 10, 6, 14}:
            return num_mlp_1_1_output == 0
        elif mlp_1_1_output in {3, 7}:
            return num_mlp_1_1_output == 11
        elif mlp_1_1_output in {4, 5}:
            return num_mlp_1_1_output == 10
        elif mlp_1_1_output in {8, 16}:
            return num_mlp_1_1_output == 12
        elif mlp_1_1_output in {9}:
            return num_mlp_1_1_output == 8
        elif mlp_1_1_output in {11}:
            return num_mlp_1_1_output == 14
        elif mlp_1_1_output in {12}:
            return num_mlp_1_1_output == 4
        elif mlp_1_1_output in {13}:
            return num_mlp_1_1_output == 15
        elif mlp_1_1_output in {15}:
            return num_mlp_1_1_output == 19
        elif mlp_1_1_output in {17}:
            return num_mlp_1_1_output == 3
        elif mlp_1_1_output in {19}:
            return num_mlp_1_1_output == 16

    attn_2_6_pattern = select_closest(
        num_mlp_1_1_outputs, mlp_1_1_outputs, predicate_2_6
    )
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_0_0_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_5_output, attn_0_6_output):
        if attn_0_5_output in {"<s>", "("}:
            return attn_0_6_output == "("
        elif attn_0_5_output in {")"}:
            return attn_0_6_output == "<s>"

    attn_2_7_pattern = select_closest(attn_0_6_outputs, attn_0_5_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, tokens)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_0_output, attn_0_1_output):
        if mlp_1_0_output in {0, 3, 4, 7, 9, 10, 11, 13, 14, 15, 18, 19}:
            return attn_0_1_output == ""
        elif mlp_1_0_output in {1, 2, 5, 6, 8, 12, 16, 17}:
            return attn_0_1_output == ")"

    num_attn_2_0_pattern = select(attn_0_1_outputs, mlp_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_0_7_output):
        if position in {0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return attn_0_7_output == ""
        elif position in {1, 9}:
            return attn_0_7_output == "("
        elif position in {7}:
            return attn_0_7_output == ")"

    num_attn_2_1_pattern = select(attn_0_7_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_3_output, attn_0_6_output):
        if attn_0_3_output in {"("}:
            return attn_0_6_output == ""
        elif attn_0_3_output in {"<s>", ")"}:
            return attn_0_6_output == ")"

    num_attn_2_2_pattern = select(attn_0_6_outputs, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_attn_1_1_output, k_attn_1_1_output):
        if q_attn_1_1_output in {"("}:
            return k_attn_1_1_output == "<pad>"
        elif q_attn_1_1_output in {"<s>", ")"}:
            return k_attn_1_1_output == ""

    num_attn_2_3_pattern = select(attn_1_1_outputs, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_0_output, num_mlp_0_0_output):
        if attn_1_0_output in {"("}:
            return num_mlp_0_0_output == 1
        elif attn_1_0_output in {")"}:
            return num_mlp_0_0_output == 14
        elif attn_1_0_output in {"<s>"}:
            return num_mlp_0_0_output == 18

    num_attn_2_4_pattern = select(
        num_mlp_0_0_outputs, attn_1_0_outputs, num_predicate_2_4
    )
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_4_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_5_output, attn_0_2_output):
        if attn_0_5_output in {"<s>", "(", ")"}:
            return attn_0_2_output == ")"

    num_attn_2_5_pattern = select(attn_0_2_outputs, attn_0_5_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_4_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_4_output, attn_1_2_output):
        if attn_1_4_output in {"(", ")"}:
            return attn_1_2_output == ""
        elif attn_1_4_output in {"<s>"}:
            return attn_1_2_output == ")"

    num_attn_2_6_pattern = select(attn_1_2_outputs, attn_1_4_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(q_attn_1_2_output, k_attn_1_2_output):
        if q_attn_1_2_output in {"<s>", "(", ")"}:
            return k_attn_1_2_output == ")"

    num_attn_2_7_pattern = select(attn_1_2_outputs, attn_1_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_1_5_output):
        key = (attn_2_2_output, attn_1_5_output)
        if key in {
            ("(", ")"),
            ("(", "<s>"),
            ("<s>", "("),
            ("<s>", ")"),
            ("<s>", "<s>"),
        }:
            return 16
        elif key in {("(", "(")}:
            return 15
        return 18

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_1_5_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, attn_1_3_output):
        key = (attn_2_1_output, attn_1_3_output)
        if key in {("(", "(")}:
            return 5
        elif key in {("<s>", "(")}:
            return 2
        elif key in {(")", "("), (")", ")"), (")", "<s>")}:
            return 11
        return 17

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_1_1_output):
        key = (num_attn_2_3_output, num_attn_1_1_output)
        return 2

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_5_output, num_attn_2_1_output):
        key = (num_attn_1_5_output, num_attn_2_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
        }:
            return 12
        return 9

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", ")", ")", "(", "(", ")", "(", ")", "(", ")"]))
