import numpy as np
import IPython

def transformToRes(responses_input):
    response_output = {}

    for id in range(responses_input.shape[0]):
        res_id = {}
        for oi in range(responses_input.shape[1]):
            if 1 in responses_input[id][oi]: # not missing
                lb = np.where(responses_input[id][oi] == 1)[0][0]  # index of the class
                res_id[oi] = lb
        if len(res_id) > 0:
            response_output[id] = res_id  # dict {instanceid: {oi: labelid}}
    return response_output

def getOracleAnsNum(responses, oracleList):
    ans_num = {oi: 0 for oi in oracleList}
    for id in responses:
        for oi in responses[id]:
            ans_num[oi] += 1
    return ans_num

def IWMV(responses_input, filter_labels, fix_answers, max_iter=500, tol=0.00001):
    '''
    label aggregation with iterative weighted majority voting
    :param responses_input: |instance| x |class| x |oracle|, elem = {0,1}, missing = -1
    :param fix_answers: a dict, {instanceid: labelid}
    :param filter_labels:
    :return:
    '''
    label2id = {filter_labels[li]: li for li in range(len(filter_labels))}
    responses_input = responses_input.swapaxes(1, 2)
    responses = transformToRes(responses_input)

    oracle_num = responses_input.shape[1]
    instance_num = responses_input.shape[0]

    ans_num = getOracleAnsNum(responses, range(oracle_num))  # {oracle id: answer num}

    vote_weight = np.ones(oracle_num)
    weight = np.zeros(oracle_num)
    predictY = {}  # id: label
    id_conf = {}
    converged = False
    old_weight = weight.copy()

    for iter in range(max_iter):
        # e-step: y^
        for idx in responses:
            if idx in fix_answers:
                predictY[idx] = fix_answers[idx]
                id_conf[idx] = 1.0
            else:
                votes = [0.0 for i in range(len(filter_labels))]
                for oi in range(oracle_num):
                    if oi in responses[idx]: # if not missing
                        ans_idx = int(responses[idx][oi])
                        votes[ans_idx] += vote_weight[oi]

                # if iter > 0 and corpusObj.m_idx2Label[votes.index(max(votes))] != predictY[idx] :
                #    print("update ans")
                confidence = max(votes) / sum(votes)
                predictY[idx] = filter_labels[votes.index(max(votes))]
                id_conf[idx] = confidence

        # m-step: w, v
        for oi in range(oracle_num):
            #weight[oi] = float(sum([1 for idx in range(instance_num) if
            #                       filter_labels[int(matrix_res[oi][idx])] == predictY[idx]])) / instance_num
            if ans_num[oi] > 0:
                weight[oi] = float(sum([1 for idx in responses if oi in responses[idx] and \
                                    filter_labels[int(responses[idx][oi])] == predictY[idx]])) / ans_num[oi]
                vote_weight[oi] = weight[oi] * len(filter_labels) - 1

        # converge
        weight_diff = np.sum(np.abs(weight - old_weight))
        if weight_diff < tol:
            converged = True
            print("converged iter: ", iter)
            break

        old_weight = weight.copy()

    # labels sorted by instance id
    predictY_key = list(predictY.keys())
    predictY_key.sort()
    predictY = np.array([predictY[k] for k in predictY_key])
    return predictY, weight, id_conf
