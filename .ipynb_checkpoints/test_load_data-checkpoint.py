# Jinwook Huh

import load_data as ld
import p3_util as ut

acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_ts	 = ld.get_imu('imuRaw20')

FR, RL, RR, RL,enc_ts = ld.get_encoder('Encoders20')

lidar = ld.get_lidar('Hokuyo20')
ut.replay_lidar(lidar)
