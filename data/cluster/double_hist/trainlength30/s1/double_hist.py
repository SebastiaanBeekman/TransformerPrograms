import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/double_hist/trainlength30/s1/double_hist_weights.csv",
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
    def predicate_0_0(q_token, k_token):
        if q_token in {"0", "1"}:
            return k_token == "3"
        elif q_token in {"3", "2"}:
            return k_token == "5"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"5", "<s>"}:
            return k_token == "4"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0", "4", "1"}:
            return k_token == "1"
        elif q_token in {"5", "2"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 8, 13, 22, 23, 28}:
            return token == "2"
        elif position in {1, 10, 20, 14}:
            return token == "3"
        elif position in {2, 36, 18, 19, 26}:
            return token == "4"
        elif position in {32, 3, 5, 15, 25}:
            return token == "1"
        elif position in {4, 16, 21, 27, 29, 30, 31}:
            return token == "5"
        elif position in {6, 9, 12, 17, 24}:
            return token == "0"
        elif position in {33, 34, 35, 37, 38, 7, 39, 11}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4, 32, 34, 38, 39, 30}:
            return k_position == 4
        elif q_position in {5, 6}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 10, 11, 12, 13, 14}:
            return k_position == 2
        elif q_position in {16, 9}:
            return k_position == 11
        elif q_position in {31, 15}:
            return k_position == 10
        elif q_position in {24, 17, 35, 20}:
            return k_position == 12
        elif q_position in {18, 26}:
            return k_position == 15
        elif q_position in {19, 21}:
            return k_position == 13
        elif q_position in {28, 22}:
            return k_position == 14
        elif q_position in {23}:
            return k_position == 16
        elif q_position in {25}:
            return k_position == 17
        elif q_position in {27}:
            return k_position == 18
        elif q_position in {29}:
            return k_position == 21
        elif q_position in {33, 37}:
            return k_position == 7
        elif q_position in {36}:
            return k_position == 37

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"2", "5", "4", "1", "3", "0", "<s>"}:
            return k_token == ""

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"2", "5", "4", "1", "3", "0", "<s>"}:
            return k_token == ""

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 16
        elif q_position in {1, 12}:
            return k_position == 21
        elif q_position in {2}:
            return k_position == 18
        elif q_position in {9, 3, 7}:
            return k_position == 20
        elif q_position in {4}:
            return k_position == 17
        elif q_position in {5}:
            return k_position == 22
        elif q_position in {11, 13, 6}:
            return k_position == 23
        elif q_position in {8}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 28
        elif q_position in {28, 14}:
            return k_position == 37
        elif q_position in {29, 30, 15}:
            return k_position == 29
        elif q_position in {16, 18}:
            return k_position == 24
        elif q_position in {17}:
            return k_position == 13
        elif q_position in {26, 19}:
            return k_position == 35
        elif q_position in {34, 20}:
            return k_position == 31
        elif q_position in {21}:
            return k_position == 33
        elif q_position in {22, 31}:
            return k_position == 27
        elif q_position in {27, 36, 23}:
            return k_position == 39
        elif q_position in {24, 38, 39}:
            return k_position == 38
        elif q_position in {25, 35}:
            return k_position == 30
        elif q_position in {32}:
            return k_position == 36
        elif q_position in {33}:
            return k_position == 26
        elif q_position in {37}:
            return k_position == 34

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output):
        key = attn_0_1_output
        if key in {"2", "4"}:
            return 2
        elif key in {"3"}:
            return 7
        return 11

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_1_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output):
        key = attn_0_1_output
        if key in {"", "2", "<s>"}:
            return 8
        return 7

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in attn_0_1_outputs]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 3

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 30

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, mlp_0_1_output):
        if token in {"0"}:
            return mlp_0_1_output == 34
        elif token in {"5", "1", "<s>"}:
            return mlp_0_1_output == 4
        elif token in {"2"}:
            return mlp_0_1_output == 39
        elif token in {"3"}:
            return mlp_0_1_output == 31
        elif token in {"4"}:
            return mlp_0_1_output == 8

    attn_1_0_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, position):
        if attn_0_3_output in {0, 2}:
            return position == 3
        elif attn_0_3_output in {1}:
            return position == 21
        elif attn_0_3_output in {35, 3, 30}:
            return position == 6
        elif attn_0_3_output in {4, 31}:
            return position == 5
        elif attn_0_3_output in {5, 7}:
            return position == 9
        elif attn_0_3_output in {6}:
            return position == 1
        elif attn_0_3_output in {8}:
            return position == 23
        elif attn_0_3_output in {9, 10, 18}:
            return position == 11
        elif attn_0_3_output in {11, 28, 15}:
            return position == 18
        elif attn_0_3_output in {12}:
            return position == 4
        elif attn_0_3_output in {13}:
            return position == 29
        elif attn_0_3_output in {19, 14}:
            return position == 22
        elif attn_0_3_output in {16, 26}:
            return position == 8
        elif attn_0_3_output in {17, 38}:
            return position == 17
        elif attn_0_3_output in {20}:
            return position == 27
        elif attn_0_3_output in {25, 33, 21, 23}:
            return position == 13
        elif attn_0_3_output in {34, 22, 39}:
            return position == 26
        elif attn_0_3_output in {24, 27}:
            return position == 28
        elif attn_0_3_output in {29}:
            return position == 25
        elif attn_0_3_output in {32}:
            return position == 10
        elif attn_0_3_output in {36}:
            return position == 34
        elif attn_0_3_output in {37}:
            return position == 39

    attn_1_1_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_3_output, position):
        if attn_0_3_output in {0, 34, 4, 38}:
            return position == 5
        elif attn_0_3_output in {1, 36, 13}:
            return position == 11
        elif attn_0_3_output in {2, 12, 21, 7}:
            return position == 8
        elif attn_0_3_output in {17, 3, 29}:
            return position == 22
        elif attn_0_3_output in {8, 5}:
            return position == 6
        elif attn_0_3_output in {6}:
            return position == 7
        elif attn_0_3_output in {9, 26, 23}:
            return position == 17
        elif attn_0_3_output in {10, 14}:
            return position == 12
        elif attn_0_3_output in {11}:
            return position == 10
        elif attn_0_3_output in {15}:
            return position == 13
        elif attn_0_3_output in {16}:
            return position == 18
        elif attn_0_3_output in {18, 39}:
            return position == 19
        elif attn_0_3_output in {32, 35, 37, 19, 28}:
            return position == 24
        elif attn_0_3_output in {20}:
            return position == 14
        elif attn_0_3_output in {22}:
            return position == 23
        elif attn_0_3_output in {24, 27}:
            return position == 21
        elif attn_0_3_output in {25}:
            return position == 26
        elif attn_0_3_output in {30}:
            return position == 25
        elif attn_0_3_output in {31}:
            return position == 36
        elif attn_0_3_output in {33}:
            return position == 4

    attn_1_2_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"3", "1"}:
            return k_token == ""
        elif q_token in {"2", "<s>"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "1"

    attn_1_3_pattern = select_closest(tokens, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, attn_0_3_output):
        if attn_0_2_output in {0, 38, 13, 14, 19, 28}:
            return attn_0_3_output == 31
        elif attn_0_2_output in {1, 4, 30, 39}:
            return attn_0_3_output == 37
        elif attn_0_2_output in {2, 37, 22, 23, 26, 29}:
            return attn_0_3_output == 36
        elif attn_0_2_output in {8, 3, 5, 6}:
            return attn_0_3_output == 35
        elif attn_0_2_output in {25, 7}:
            return attn_0_3_output == 38
        elif attn_0_2_output in {32, 36, 9, 15, 16, 17, 21, 24}:
            return attn_0_3_output == 30
        elif attn_0_2_output in {10, 20, 34}:
            return attn_0_3_output == 34
        elif attn_0_2_output in {11, 12}:
            return attn_0_3_output == 32
        elif attn_0_2_output in {33, 18, 35}:
            return attn_0_3_output == 33
        elif attn_0_2_output in {27}:
            return attn_0_3_output == 27
        elif attn_0_2_output in {31}:
            return attn_0_3_output == 39

    num_attn_1_0_pattern = select(attn_0_3_outputs, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"2", "5", "4", "3", "0"}:
            return k_attn_0_1_output == ""
        elif q_attn_0_1_output in {"1"}:
            return k_attn_0_1_output == "<pad>"
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == "<s>"

    num_attn_1_1_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, token):
        if attn_0_1_output in {"2", "5", "4", "1", "3", "0", "<s>"}:
            return token == ""

    num_attn_1_2_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(token, attn_0_1_output):
        if token in {"2", "5", "4", "1", "3", "0", "<s>"}:
            return attn_0_1_output == ""

    num_attn_1_3_pattern = select(attn_0_1_outputs, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output):
        key = attn_1_0_output
        if key in {24, 27}:
            return 13
        return 7

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_1_0_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_0_output, attn_1_3_output):
        key = (mlp_0_0_output, attn_1_3_output)
        return 30

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_0_0_output):
        key = (num_attn_1_3_output, num_attn_0_0_output)
        return 0

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output, num_attn_1_2_output):
        key = (num_attn_0_3_output, num_attn_1_2_output)
        if key in {
            (6, 0),
            (7, 0),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
            (10, 0),
            (10, 1),
            (10, 2),
            (11, 0),
            (11, 1),
            (11, 2),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (31, 0),
            (31, 1),
            (31, 2),
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
            (32, 0),
            (32, 1),
            (32, 2),
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
            (33, 0),
            (33, 1),
            (33, 2),
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
            (34, 0),
            (34, 1),
            (34, 2),
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
            (35, 0),
            (35, 1),
            (35, 2),
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
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
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
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
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
            (37, 15),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
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
            (38, 15),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
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
            (39, 15),
            (39, 16),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (40, 10),
            (40, 11),
            (40, 12),
            (40, 13),
            (40, 14),
            (40, 15),
            (40, 16),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (41, 10),
            (41, 11),
            (41, 12),
            (41, 13),
            (41, 14),
            (41, 15),
            (41, 16),
            (41, 17),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (42, 10),
            (42, 11),
            (42, 12),
            (42, 13),
            (42, 14),
            (42, 15),
            (42, 16),
            (42, 17),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (43, 10),
            (43, 11),
            (43, 12),
            (43, 13),
            (43, 14),
            (43, 15),
            (43, 16),
            (43, 17),
            (43, 18),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (44, 10),
            (44, 11),
            (44, 12),
            (44, 13),
            (44, 14),
            (44, 15),
            (44, 16),
            (44, 17),
            (44, 18),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (45, 10),
            (45, 11),
            (45, 12),
            (45, 13),
            (45, 14),
            (45, 15),
            (45, 16),
            (45, 17),
            (45, 18),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
            (46, 11),
            (46, 12),
            (46, 13),
            (46, 14),
            (46, 15),
            (46, 16),
            (46, 17),
            (46, 18),
            (46, 19),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (47, 10),
            (47, 11),
            (47, 12),
            (47, 13),
            (47, 14),
            (47, 15),
            (47, 16),
            (47, 17),
            (47, 18),
            (47, 19),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 7),
            (48, 8),
            (48, 9),
            (48, 10),
            (48, 11),
            (48, 12),
            (48, 13),
            (48, 14),
            (48, 15),
            (48, 16),
            (48, 17),
            (48, 18),
            (48, 19),
            (48, 20),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (49, 7),
            (49, 8),
            (49, 9),
            (49, 10),
            (49, 11),
            (49, 12),
            (49, 13),
            (49, 14),
            (49, 15),
            (49, 16),
            (49, 17),
            (49, 18),
            (49, 19),
            (49, 20),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 7),
            (50, 8),
            (50, 9),
            (50, 10),
            (50, 11),
            (50, 12),
            (50, 13),
            (50, 14),
            (50, 15),
            (50, 16),
            (50, 17),
            (50, 18),
            (50, 19),
            (50, 20),
            (50, 21),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (51, 7),
            (51, 8),
            (51, 9),
            (51, 10),
            (51, 11),
            (51, 12),
            (51, 13),
            (51, 14),
            (51, 15),
            (51, 16),
            (51, 17),
            (51, 18),
            (51, 19),
            (51, 20),
            (51, 21),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (52, 7),
            (52, 8),
            (52, 9),
            (52, 10),
            (52, 11),
            (52, 12),
            (52, 13),
            (52, 14),
            (52, 15),
            (52, 16),
            (52, 17),
            (52, 18),
            (52, 19),
            (52, 20),
            (52, 21),
            (52, 22),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (53, 7),
            (53, 8),
            (53, 9),
            (53, 10),
            (53, 11),
            (53, 12),
            (53, 13),
            (53, 14),
            (53, 15),
            (53, 16),
            (53, 17),
            (53, 18),
            (53, 19),
            (53, 20),
            (53, 21),
            (53, 22),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (54, 7),
            (54, 8),
            (54, 9),
            (54, 10),
            (54, 11),
            (54, 12),
            (54, 13),
            (54, 14),
            (54, 15),
            (54, 16),
            (54, 17),
            (54, 18),
            (54, 19),
            (54, 20),
            (54, 21),
            (54, 22),
            (54, 23),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (55, 7),
            (55, 8),
            (55, 9),
            (55, 10),
            (55, 11),
            (55, 12),
            (55, 13),
            (55, 14),
            (55, 15),
            (55, 16),
            (55, 17),
            (55, 18),
            (55, 19),
            (55, 20),
            (55, 21),
            (55, 22),
            (55, 23),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (56, 7),
            (56, 8),
            (56, 9),
            (56, 10),
            (56, 11),
            (56, 12),
            (56, 13),
            (56, 14),
            (56, 15),
            (56, 16),
            (56, 17),
            (56, 18),
            (56, 19),
            (56, 20),
            (56, 21),
            (56, 22),
            (56, 23),
            (56, 24),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (57, 8),
            (57, 9),
            (57, 10),
            (57, 11),
            (57, 12),
            (57, 13),
            (57, 14),
            (57, 15),
            (57, 16),
            (57, 17),
            (57, 18),
            (57, 19),
            (57, 20),
            (57, 21),
            (57, 22),
            (57, 23),
            (57, 24),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (58, 8),
            (58, 9),
            (58, 10),
            (58, 11),
            (58, 12),
            (58, 13),
            (58, 14),
            (58, 15),
            (58, 16),
            (58, 17),
            (58, 18),
            (58, 19),
            (58, 20),
            (58, 21),
            (58, 22),
            (58, 23),
            (58, 24),
            (58, 25),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (59, 8),
            (59, 9),
            (59, 10),
            (59, 11),
            (59, 12),
            (59, 13),
            (59, 14),
            (59, 15),
            (59, 16),
            (59, 17),
            (59, 18),
            (59, 19),
            (59, 20),
            (59, 21),
            (59, 22),
            (59, 23),
            (59, 24),
            (59, 25),
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (60, 6),
            (60, 7),
            (60, 8),
            (60, 9),
            (60, 10),
            (60, 11),
            (60, 12),
            (60, 13),
            (60, 14),
            (60, 15),
            (60, 16),
            (60, 17),
            (60, 18),
            (60, 19),
            (60, 20),
            (60, 21),
            (60, 22),
            (60, 23),
            (60, 24),
            (60, 25),
            (60, 26),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (61, 7),
            (61, 8),
            (61, 9),
            (61, 10),
            (61, 11),
            (61, 12),
            (61, 13),
            (61, 14),
            (61, 15),
            (61, 16),
            (61, 17),
            (61, 18),
            (61, 19),
            (61, 20),
            (61, 21),
            (61, 22),
            (61, 23),
            (61, 24),
            (61, 25),
            (61, 26),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (62, 7),
            (62, 8),
            (62, 9),
            (62, 10),
            (62, 11),
            (62, 12),
            (62, 13),
            (62, 14),
            (62, 15),
            (62, 16),
            (62, 17),
            (62, 18),
            (62, 19),
            (62, 20),
            (62, 21),
            (62, 22),
            (62, 23),
            (62, 24),
            (62, 25),
            (62, 26),
            (62, 27),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (63, 7),
            (63, 8),
            (63, 9),
            (63, 10),
            (63, 11),
            (63, 12),
            (63, 13),
            (63, 14),
            (63, 15),
            (63, 16),
            (63, 17),
            (63, 18),
            (63, 19),
            (63, 20),
            (63, 21),
            (63, 22),
            (63, 23),
            (63, 24),
            (63, 25),
            (63, 26),
            (63, 27),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (64, 7),
            (64, 8),
            (64, 9),
            (64, 10),
            (64, 11),
            (64, 12),
            (64, 13),
            (64, 14),
            (64, 15),
            (64, 16),
            (64, 17),
            (64, 18),
            (64, 19),
            (64, 20),
            (64, 21),
            (64, 22),
            (64, 23),
            (64, 24),
            (64, 25),
            (64, 26),
            (64, 27),
            (64, 28),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (65, 7),
            (65, 8),
            (65, 9),
            (65, 10),
            (65, 11),
            (65, 12),
            (65, 13),
            (65, 14),
            (65, 15),
            (65, 16),
            (65, 17),
            (65, 18),
            (65, 19),
            (65, 20),
            (65, 21),
            (65, 22),
            (65, 23),
            (65, 24),
            (65, 25),
            (65, 26),
            (65, 27),
            (65, 28),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (66, 7),
            (66, 8),
            (66, 9),
            (66, 10),
            (66, 11),
            (66, 12),
            (66, 13),
            (66, 14),
            (66, 15),
            (66, 16),
            (66, 17),
            (66, 18),
            (66, 19),
            (66, 20),
            (66, 21),
            (66, 22),
            (66, 23),
            (66, 24),
            (66, 25),
            (66, 26),
            (66, 27),
            (66, 28),
            (66, 29),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (67, 7),
            (67, 8),
            (67, 9),
            (67, 10),
            (67, 11),
            (67, 12),
            (67, 13),
            (67, 14),
            (67, 15),
            (67, 16),
            (67, 17),
            (67, 18),
            (67, 19),
            (67, 20),
            (67, 21),
            (67, 22),
            (67, 23),
            (67, 24),
            (67, 25),
            (67, 26),
            (67, 27),
            (67, 28),
            (67, 29),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (68, 7),
            (68, 8),
            (68, 9),
            (68, 10),
            (68, 11),
            (68, 12),
            (68, 13),
            (68, 14),
            (68, 15),
            (68, 16),
            (68, 17),
            (68, 18),
            (68, 19),
            (68, 20),
            (68, 21),
            (68, 22),
            (68, 23),
            (68, 24),
            (68, 25),
            (68, 26),
            (68, 27),
            (68, 28),
            (68, 29),
            (68, 30),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (69, 8),
            (69, 9),
            (69, 10),
            (69, 11),
            (69, 12),
            (69, 13),
            (69, 14),
            (69, 15),
            (69, 16),
            (69, 17),
            (69, 18),
            (69, 19),
            (69, 20),
            (69, 21),
            (69, 22),
            (69, 23),
            (69, 24),
            (69, 25),
            (69, 26),
            (69, 27),
            (69, 28),
            (69, 29),
            (69, 30),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (70, 8),
            (70, 9),
            (70, 10),
            (70, 11),
            (70, 12),
            (70, 13),
            (70, 14),
            (70, 15),
            (70, 16),
            (70, 17),
            (70, 18),
            (70, 19),
            (70, 20),
            (70, 21),
            (70, 22),
            (70, 23),
            (70, 24),
            (70, 25),
            (70, 26),
            (70, 27),
            (70, 28),
            (70, 29),
            (70, 30),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (71, 8),
            (71, 9),
            (71, 10),
            (71, 11),
            (71, 12),
            (71, 13),
            (71, 14),
            (71, 15),
            (71, 16),
            (71, 17),
            (71, 18),
            (71, 19),
            (71, 20),
            (71, 21),
            (71, 22),
            (71, 23),
            (71, 24),
            (71, 25),
            (71, 26),
            (71, 27),
            (71, 28),
            (71, 29),
            (71, 30),
            (71, 31),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (72, 8),
            (72, 9),
            (72, 10),
            (72, 11),
            (72, 12),
            (72, 13),
            (72, 14),
            (72, 15),
            (72, 16),
            (72, 17),
            (72, 18),
            (72, 19),
            (72, 20),
            (72, 21),
            (72, 22),
            (72, 23),
            (72, 24),
            (72, 25),
            (72, 26),
            (72, 27),
            (72, 28),
            (72, 29),
            (72, 30),
            (72, 31),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (73, 8),
            (73, 9),
            (73, 10),
            (73, 11),
            (73, 12),
            (73, 13),
            (73, 14),
            (73, 15),
            (73, 16),
            (73, 17),
            (73, 18),
            (73, 19),
            (73, 20),
            (73, 21),
            (73, 22),
            (73, 23),
            (73, 24),
            (73, 25),
            (73, 26),
            (73, 27),
            (73, 28),
            (73, 29),
            (73, 30),
            (73, 31),
            (73, 32),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (74, 8),
            (74, 9),
            (74, 10),
            (74, 11),
            (74, 12),
            (74, 13),
            (74, 14),
            (74, 15),
            (74, 16),
            (74, 17),
            (74, 18),
            (74, 19),
            (74, 20),
            (74, 21),
            (74, 22),
            (74, 23),
            (74, 24),
            (74, 25),
            (74, 26),
            (74, 27),
            (74, 28),
            (74, 29),
            (74, 30),
            (74, 31),
            (74, 32),
            (75, 0),
            (75, 1),
            (75, 2),
            (75, 3),
            (75, 4),
            (75, 5),
            (75, 6),
            (75, 7),
            (75, 8),
            (75, 9),
            (75, 10),
            (75, 11),
            (75, 12),
            (75, 13),
            (75, 14),
            (75, 15),
            (75, 16),
            (75, 17),
            (75, 18),
            (75, 19),
            (75, 20),
            (75, 21),
            (75, 22),
            (75, 23),
            (75, 24),
            (75, 25),
            (75, 26),
            (75, 27),
            (75, 28),
            (75, 29),
            (75, 30),
            (75, 31),
            (75, 32),
            (75, 33),
            (76, 0),
            (76, 1),
            (76, 2),
            (76, 3),
            (76, 4),
            (76, 5),
            (76, 6),
            (76, 7),
            (76, 8),
            (76, 9),
            (76, 10),
            (76, 11),
            (76, 12),
            (76, 13),
            (76, 14),
            (76, 15),
            (76, 16),
            (76, 17),
            (76, 18),
            (76, 19),
            (76, 20),
            (76, 21),
            (76, 22),
            (76, 23),
            (76, 24),
            (76, 25),
            (76, 26),
            (76, 27),
            (76, 28),
            (76, 29),
            (76, 30),
            (76, 31),
            (76, 32),
            (76, 33),
            (77, 0),
            (77, 1),
            (77, 2),
            (77, 3),
            (77, 4),
            (77, 5),
            (77, 6),
            (77, 7),
            (77, 8),
            (77, 9),
            (77, 10),
            (77, 11),
            (77, 12),
            (77, 13),
            (77, 14),
            (77, 15),
            (77, 16),
            (77, 17),
            (77, 18),
            (77, 19),
            (77, 20),
            (77, 21),
            (77, 22),
            (77, 23),
            (77, 24),
            (77, 25),
            (77, 26),
            (77, 27),
            (77, 28),
            (77, 29),
            (77, 30),
            (77, 31),
            (77, 32),
            (77, 33),
            (77, 34),
            (78, 0),
            (78, 1),
            (78, 2),
            (78, 3),
            (78, 4),
            (78, 5),
            (78, 6),
            (78, 7),
            (78, 8),
            (78, 9),
            (78, 10),
            (78, 11),
            (78, 12),
            (78, 13),
            (78, 14),
            (78, 15),
            (78, 16),
            (78, 17),
            (78, 18),
            (78, 19),
            (78, 20),
            (78, 21),
            (78, 22),
            (78, 23),
            (78, 24),
            (78, 25),
            (78, 26),
            (78, 27),
            (78, 28),
            (78, 29),
            (78, 30),
            (78, 31),
            (78, 32),
            (78, 33),
            (78, 34),
            (79, 0),
            (79, 1),
            (79, 2),
            (79, 3),
            (79, 4),
            (79, 5),
            (79, 6),
            (79, 7),
            (79, 8),
            (79, 9),
            (79, 10),
            (79, 11),
            (79, 12),
            (79, 13),
            (79, 14),
            (79, 15),
            (79, 16),
            (79, 17),
            (79, 18),
            (79, 19),
            (79, 20),
            (79, 21),
            (79, 22),
            (79, 23),
            (79, 24),
            (79, 25),
            (79, 26),
            (79, 27),
            (79, 28),
            (79, 29),
            (79, 30),
            (79, 31),
            (79, 32),
            (79, 33),
            (79, 34),
            (79, 35),
        }:
            return 18
        elif key in {
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 66),
            (0, 67),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (1, 60),
            (1, 61),
            (1, 62),
            (1, 63),
            (1, 64),
            (1, 65),
            (1, 66),
            (1, 67),
            (1, 68),
            (1, 69),
            (1, 70),
            (1, 71),
            (1, 72),
            (1, 73),
            (1, 74),
            (1, 75),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (2, 60),
            (2, 61),
            (2, 62),
            (2, 63),
            (2, 64),
            (2, 65),
            (2, 66),
            (2, 67),
            (2, 68),
            (2, 69),
            (2, 70),
            (2, 71),
            (2, 72),
            (2, 73),
            (2, 74),
            (2, 75),
            (2, 76),
            (2, 77),
            (2, 78),
            (2, 79),
            (3, 61),
            (3, 62),
            (3, 63),
            (3, 64),
            (3, 65),
            (3, 66),
            (3, 67),
            (3, 68),
            (3, 69),
            (3, 70),
            (3, 71),
            (3, 72),
            (3, 73),
            (3, 74),
            (3, 75),
            (3, 76),
            (3, 77),
            (3, 78),
            (3, 79),
            (4, 69),
            (4, 70),
            (4, 71),
            (4, 72),
            (4, 73),
            (4, 74),
            (4, 75),
            (4, 76),
            (4, 77),
            (4, 78),
            (4, 79),
            (5, 77),
            (5, 78),
            (5, 79),
        }:
            return 0
        elif key in {
            (0, 68),
            (0, 69),
            (0, 70),
            (0, 71),
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (0, 77),
            (0, 78),
            (0, 79),
            (1, 76),
            (1, 77),
            (1, 78),
            (1, 79),
        }:
            return 30
        return 20

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, mlp_0_0_output):
        if token in {"0", "4"}:
            return mlp_0_0_output == 32
        elif token in {"1"}:
            return mlp_0_0_output == 2
        elif token in {"2"}:
            return mlp_0_0_output == 26
        elif token in {"3"}:
            return mlp_0_0_output == 4
        elif token in {"5"}:
            return mlp_0_0_output == 6
        elif token in {"<s>"}:
            return mlp_0_0_output == 29

    attn_2_0_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 17}:
            return k_position == 20
        elif q_position in {1, 2, 3, 4, 33, 34, 35, 36, 37, 38, 12, 23, 30, 31}:
            return k_position == 6
        elif q_position in {5, 7, 8, 9, 11, 18}:
            return k_position == 10
        elif q_position in {29, 6, 15}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 25
        elif q_position in {16}:
            return k_position == 9
        elif q_position in {19}:
            return k_position == 23
        elif q_position in {20, 39}:
            return k_position == 16
        elif q_position in {21}:
            return k_position == 15
        elif q_position in {24, 25, 22}:
            return k_position == 8
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {27}:
            return k_position == 14
        elif q_position in {28}:
            return k_position == 22
        elif q_position in {32}:
            return k_position == 5

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_2_output, attn_0_2_output):
        if attn_1_2_output in {0, 5}:
            return attn_0_2_output == 8
        elif attn_1_2_output in {1, 38}:
            return attn_0_2_output == 5
        elif attn_1_2_output in {2}:
            return attn_0_2_output == 17
        elif attn_1_2_output in {3, 7}:
            return attn_0_2_output == 18
        elif attn_1_2_output in {4}:
            return attn_0_2_output == 23
        elif attn_1_2_output in {34, 36, 37, 6}:
            return attn_0_2_output == 0
        elif attn_1_2_output in {8, 21, 23}:
            return attn_0_2_output == 16
        elif attn_1_2_output in {24, 9}:
            return attn_0_2_output == 20
        elif attn_1_2_output in {10, 35}:
            return attn_0_2_output == 22
        elif attn_1_2_output in {11}:
            return attn_0_2_output == 28
        elif attn_1_2_output in {33, 18, 12}:
            return attn_0_2_output == 6
        elif attn_1_2_output in {13}:
            return attn_0_2_output == 2
        elif attn_1_2_output in {32, 14, 16, 17, 20}:
            return attn_0_2_output == 10
        elif attn_1_2_output in {19, 15}:
            return attn_0_2_output == 12
        elif attn_1_2_output in {22}:
            return attn_0_2_output == 11
        elif attn_1_2_output in {25, 26, 28}:
            return attn_0_2_output == 13
        elif attn_1_2_output in {27, 30}:
            return attn_0_2_output == 1
        elif attn_1_2_output in {29}:
            return attn_0_2_output == 15
        elif attn_1_2_output in {31}:
            return attn_0_2_output == 21
        elif attn_1_2_output in {39}:
            return attn_0_2_output == 26

    attn_2_2_pattern = select_closest(attn_0_2_outputs, attn_1_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"5", "4", "0", "2"}:
            return k_attn_0_1_output == "4"
        elif q_attn_0_1_output in {"1"}:
            return k_attn_0_1_output == "<s>"
        elif q_attn_0_1_output in {"3"}:
            return k_attn_0_1_output == ""
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == "1"

    attn_2_3_pattern = select_closest(attn_0_1_outputs, attn_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, attn_0_2_output):
        if attn_1_0_output in {0}:
            return attn_0_2_output == 3
        elif attn_1_0_output in {1, 2, 4, 7}:
            return attn_0_2_output == 0
        elif attn_1_0_output in {3}:
            return attn_0_2_output == 2
        elif attn_1_0_output in {5}:
            return attn_0_2_output == 5
        elif attn_1_0_output in {6}:
            return attn_0_2_output == 6
        elif attn_1_0_output in {8}:
            return attn_0_2_output == 8
        elif attn_1_0_output in {9, 11}:
            return attn_0_2_output == 9
        elif attn_1_0_output in {10, 14}:
            return attn_0_2_output == 10
        elif attn_1_0_output in {12, 20}:
            return attn_0_2_output == 13
        elif attn_1_0_output in {19, 13, 15}:
            return attn_0_2_output == 11
        elif attn_1_0_output in {16}:
            return attn_0_2_output == 16
        elif attn_1_0_output in {17}:
            return attn_0_2_output == 15
        elif attn_1_0_output in {18, 22}:
            return attn_0_2_output == 17
        elif attn_1_0_output in {21}:
            return attn_0_2_output == 21
        elif attn_1_0_output in {28, 23}:
            return attn_0_2_output == 22
        elif attn_1_0_output in {24}:
            return attn_0_2_output == 24
        elif attn_1_0_output in {25}:
            return attn_0_2_output == 25
        elif attn_1_0_output in {26}:
            return attn_0_2_output == 26
        elif attn_1_0_output in {27}:
            return attn_0_2_output == 12
        elif attn_1_0_output in {29}:
            return attn_0_2_output == 27
        elif attn_1_0_output in {33, 36, 37, 38, 39, 30, 31}:
            return attn_0_2_output == 4
        elif attn_1_0_output in {32, 35}:
            return attn_0_2_output == 1
        elif attn_1_0_output in {34}:
            return attn_0_2_output == 7

    num_attn_2_0_pattern = select(attn_0_2_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_2_output, attn_0_3_output):
        if attn_1_2_output in {0, 34, 36, 39, 12}:
            return attn_0_3_output == 34
        elif attn_1_2_output in {1, 30, 7}:
            return attn_0_3_output == 37
        elif attn_1_2_output in {2, 5, 6, 9, 17, 19, 22, 25, 31}:
            return attn_0_3_output == 30
        elif attn_1_2_output in {32, 3, 38, 13, 15, 20, 28}:
            return attn_0_3_output == 39
        elif attn_1_2_output in {4}:
            return attn_0_3_output == 4
        elif attn_1_2_output in {8, 14}:
            return attn_0_3_output == 35
        elif attn_1_2_output in {24, 10, 37}:
            return attn_0_3_output == 33
        elif attn_1_2_output in {11, 23}:
            return attn_0_3_output == 36
        elif attn_1_2_output in {35, 16, 21, 26, 27}:
            return attn_0_3_output == 32
        elif attn_1_2_output in {18}:
            return attn_0_3_output == 31
        elif attn_1_2_output in {33, 29}:
            return attn_0_3_output == 38

    num_attn_2_1_pattern = select(attn_0_3_outputs, attn_1_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 32, 35, 19, 26}:
            return position == 34
        elif mlp_0_1_output in {1, 33, 4, 7, 16, 22, 23}:
            return position == 27
        elif mlp_0_1_output in {2, 15}:
            return position == 28
        elif mlp_0_1_output in {3}:
            return position == 31
        elif mlp_0_1_output in {5}:
            return position == 22
        elif mlp_0_1_output in {6, 21, 24, 27, 28}:
            return position == 36
        elif mlp_0_1_output in {8, 25, 20}:
            return position == 35
        elif mlp_0_1_output in {9}:
            return position == 33
        elif mlp_0_1_output in {36, 10, 13, 14, 18, 30}:
            return position == 32
        elif mlp_0_1_output in {11}:
            return position == 26
        elif mlp_0_1_output in {17, 12}:
            return position == 37
        elif mlp_0_1_output in {34, 37, 29}:
            return position == 30
        elif mlp_0_1_output in {31}:
            return position == 12
        elif mlp_0_1_output in {38}:
            return position == 23
        elif mlp_0_1_output in {39}:
            return position == 17

    num_attn_2_2_pattern = select(positions, mlp_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_1_output, token):
        if attn_0_1_output in {"2", "5", "4", "1", "3", "0"}:
            return token == ""
        elif attn_0_1_output in {"<s>"}:
            return token == "<pad>"

    num_attn_2_3_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, mlp_0_0_output):
        key = (attn_2_3_output, mlp_0_0_output)
        if key in {
            (0, 5),
            (0, 6),
            (0, 8),
            (0, 9),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 18),
            (0, 21),
            (0, 26),
            (0, 27),
            (0, 32),
            (0, 33),
            (0, 35),
            (0, 38),
            (0, 39),
            (1, 8),
            (1, 13),
            (1, 32),
            (2, 8),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 16),
            (2, 18),
            (2, 26),
            (2, 32),
            (2, 35),
            (2, 38),
            (3, 8),
            (3, 13),
            (3, 14),
            (3, 18),
            (3, 20),
            (3, 26),
            (3, 32),
            (4, 8),
            (4, 13),
            (4, 14),
            (4, 18),
            (4, 20),
            (4, 22),
            (4, 26),
            (4, 32),
            (5, 13),
            (5, 14),
            (5, 18),
            (5, 26),
            (5, 32),
            (6, 8),
            (6, 13),
            (6, 32),
            (7, 8),
            (7, 13),
            (7, 14),
            (7, 16),
            (7, 18),
            (7, 20),
            (7, 22),
            (7, 26),
            (7, 32),
            (8, 5),
            (8, 6),
            (8, 8),
            (8, 9),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 18),
            (8, 20),
            (8, 21),
            (8, 26),
            (8, 27),
            (8, 32),
            (8, 33),
            (8, 35),
            (8, 38),
            (8, 39),
            (9, 8),
            (9, 13),
            (9, 16),
            (9, 32),
            (10, 13),
            (10, 32),
            (11, 0),
            (11, 1),
            (11, 3),
            (11, 4),
            (11, 9),
            (11, 10),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 24),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (12, 13),
            (13, 8),
            (13, 13),
            (13, 14),
            (13, 18),
            (13, 26),
            (13, 32),
            (14, 8),
            (14, 13),
            (14, 14),
            (14, 18),
            (14, 26),
            (14, 32),
            (15, 8),
            (15, 13),
            (15, 14),
            (15, 18),
            (15, 26),
            (15, 32),
            (16, 3),
            (16, 8),
            (16, 13),
            (16, 14),
            (16, 18),
            (16, 20),
            (16, 22),
            (16, 26),
            (16, 32),
            (18, 14),
            (18, 18),
            (18, 20),
            (18, 22),
            (18, 26),
            (19, 13),
            (20, 8),
            (20, 13),
            (20, 32),
            (21, 3),
            (21, 8),
            (21, 13),
            (21, 14),
            (21, 18),
            (21, 20),
            (21, 22),
            (21, 26),
            (21, 32),
            (22, 8),
            (22, 13),
            (22, 14),
            (22, 16),
            (22, 18),
            (22, 20),
            (22, 22),
            (22, 26),
            (22, 32),
            (24, 8),
            (24, 13),
            (24, 14),
            (24, 18),
            (24, 26),
            (24, 32),
            (26, 3),
            (26, 5),
            (26, 6),
            (26, 8),
            (26, 9),
            (26, 12),
            (26, 13),
            (26, 14),
            (26, 15),
            (26, 16),
            (26, 17),
            (26, 19),
            (26, 21),
            (26, 23),
            (26, 25),
            (26, 26),
            (26, 27),
            (26, 29),
            (26, 32),
            (26, 33),
            (26, 35),
            (26, 36),
            (26, 37),
            (26, 38),
            (26, 39),
            (27, 5),
            (27, 6),
            (27, 8),
            (27, 9),
            (27, 12),
            (27, 13),
            (27, 14),
            (27, 15),
            (27, 16),
            (27, 18),
            (27, 20),
            (27, 21),
            (27, 22),
            (27, 23),
            (27, 26),
            (27, 27),
            (27, 32),
            (27, 33),
            (27, 35),
            (27, 36),
            (27, 37),
            (27, 38),
            (27, 39),
            (28, 3),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 12),
            (28, 13),
            (28, 14),
            (28, 16),
            (28, 17),
            (28, 18),
            (28, 20),
            (28, 22),
            (28, 25),
            (28, 26),
            (28, 28),
            (28, 29),
            (28, 32),
            (28, 35),
            (28, 36),
            (29, 8),
            (29, 13),
            (29, 32),
            (30, 8),
            (30, 13),
            (30, 18),
            (30, 26),
            (30, 32),
            (31, 8),
            (31, 12),
            (31, 13),
            (31, 16),
            (31, 18),
            (31, 22),
            (31, 26),
            (31, 32),
            (31, 35),
            (32, 13),
            (32, 14),
            (32, 18),
            (32, 20),
            (32, 22),
            (32, 26),
            (32, 32),
            (33, 8),
            (33, 13),
            (33, 14),
            (33, 16),
            (33, 18),
            (33, 20),
            (33, 26),
            (33, 32),
            (34, 8),
            (34, 13),
            (34, 14),
            (34, 16),
            (34, 18),
            (34, 24),
            (34, 26),
            (34, 29),
            (34, 31),
            (34, 32),
            (35, 5),
            (35, 6),
            (35, 8),
            (35, 9),
            (35, 12),
            (35, 13),
            (35, 14),
            (35, 15),
            (35, 16),
            (35, 18),
            (35, 20),
            (35, 26),
            (35, 27),
            (35, 32),
            (35, 35),
            (35, 38),
            (35, 39),
            (36, 5),
            (36, 6),
            (36, 8),
            (36, 9),
            (36, 12),
            (36, 13),
            (36, 15),
            (36, 16),
            (36, 21),
            (36, 26),
            (36, 27),
            (36, 32),
            (36, 33),
            (36, 35),
            (36, 38),
            (36, 39),
            (37, 5),
            (37, 6),
            (37, 8),
            (37, 9),
            (37, 12),
            (37, 13),
            (37, 14),
            (37, 15),
            (37, 16),
            (37, 18),
            (37, 21),
            (37, 23),
            (37, 26),
            (37, 27),
            (37, 32),
            (37, 33),
            (37, 35),
            (37, 36),
            (37, 37),
            (37, 38),
            (37, 39),
            (38, 8),
            (38, 13),
            (38, 16),
            (38, 29),
            (38, 32),
            (39, 13),
            (39, 14),
            (39, 18),
            (39, 26),
            (39, 32),
        }:
            return 8
        elif key in {
            (18, 29),
            (22, 29),
            (22, 38),
            (27, 29),
            (31, 7),
            (31, 10),
            (31, 11),
            (31, 14),
            (31, 17),
            (31, 21),
            (31, 23),
            (31, 30),
            (34, 38),
            (38, 38),
        }:
            return 35
        elif key in {
            (31, 0),
            (31, 1),
            (31, 4),
            (31, 20),
            (31, 24),
            (31, 29),
            (31, 34),
            (31, 36),
            (31, 38),
            (31, 39),
        }:
            return 31
        return 6

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, mlp_0_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_1_output, mlp_1_1_output):
        key = (num_mlp_0_1_output, mlp_1_1_output)
        return 39

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, mlp_1_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_3_output, num_attn_1_1_output):
        key = (num_attn_0_3_output, num_attn_1_1_output)
        return 2

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_2_0_output):
        key = (num_attn_1_1_output, num_attn_2_0_output)
        return 18

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_0_outputs)
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
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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


print(run(["<s>", "3", "4", "0", "1", "3", "5"]))
