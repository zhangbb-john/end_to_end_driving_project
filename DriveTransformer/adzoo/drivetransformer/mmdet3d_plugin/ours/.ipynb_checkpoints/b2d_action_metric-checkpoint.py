def get_action(index):
	Discrete_Actions_DICT = {
		0:  (0, 0, 1, False),
		1:  (0.7, -0.5, 0, False),
		2:  (0.7, -0.3, 0, False),
		3:  (0.7, -0.2, 0, False),
		4:  (0.7, -0.1, 0, False),
		5:  (0.7, 0, 0, False),
		6:  (0.7, 0.1, 0, False),
		7:  (0.7, 0.2, 0, False),
		8:  (0.7, 0.3, 0, False),
		9:  (0.7, 0.5, 0, False),
		10: (0.3, -0.7, 0, False),
		11: (0.3, -0.5, 0, False),
		12: (0.3, -0.3, 0, False),
		13: (0.3, -0.2, 0, False),
		14: (0.3, -0.1, 0, False),
		15: (0.3, 0, 0, False),
		16: (0.3, 0.1, 0, False),
		17: (0.3, 0.2, 0, False),
		18: (0.3, 0.3, 0, False),
		19: (0.3, 0.5, 0, False),
		20: (0.3, 0.7, 0, False),
		21: (0, -1, 0, False),
		22: (0, -0.6, 0, False),
		23: (0, -0.3, 0, False),
		24: (0, -0.1, 0, False),
		25: (1, 0, 0, False),
		26: (0, 0.1, 0, False),
		27: (0, 0.3, 0, False),
		28: (0, 0.6, 0, False),
		29: (0, 1.0, 0, False),
		30: (0.5, -0.5, 0, True),
		31: (0.5, -0.3, 0, True),
		32: (0.5, -0.2, 0, True),
		33: (0.5, -0.1, 0, True),
		34: (0.5, 0, 0, True),
		35: (0.5, 0.1, 0, True),
		36: (0.5, 0.2, 0, True),
		37: (0.5, 0.3, 0, True),
		38: (0.5, 0.5, 0, True),
		}
	throttle, steer, brake, reverse = Discrete_Actions_DICT[index]
	return throttle, steer, brake

def get_action_remaped(index):
	Discrete_Actions_DICT = {
		0: (1.0, 0.0, 0.0, False),
		1: (0.0, 0.0, 1.0, False),
		2: (0.0, 0.0, 0.0, False),
		3: (0.0, 0.0, 0.0, False),
		}
	# Discrete_Actions_DICT = {
	# 	0:  (0,   0, 1, False),
	# 	1:  (0.7, 0, 0, False),
	# 	2:  (0.3, 0, 0, False),
	# 	3:  (0,   0, 0, False),
	# 	4:  (1,   0, 0, False),
	# 	5:  (0.5, 0, 0, False),
	# 	}
	throttle, steer, brake, reverse = Discrete_Actions_DICT[index]
	return throttle, steer, brake

def compute_action_metrics(pred_command,gt_command,remaped=False):
	pred_command = pred_command.item()
	gt_command = gt_command.item()
	metirc_dict = {}
	metirc_dict['action_acc'] = 1 if pred_command==gt_command else 0
	if gt_command!=3:
		metirc_dict['cmd_valid'] = True
	else:
		metirc_dict['cmd_valid'] = False
	if remaped:
		pred_throttle, pred_steer, pred_brake = get_action_remaped(pred_command)
		gt_throttle, gt_steer, gt_brake = get_action_remaped(gt_command)        
	else:
		pred_throttle, pred_steer, pred_brake = get_action(pred_command)
		gt_throttle, gt_steer, gt_brake = get_action(gt_command)
	metirc_dict['throttle_L1'] = abs(pred_throttle-gt_throttle)
	metirc_dict['steer_L1'] = abs(pred_steer-gt_steer)
	metirc_dict['brake_L1'] = abs(pred_brake-gt_brake)
	return metirc_dict
