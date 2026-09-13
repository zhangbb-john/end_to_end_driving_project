# 4090 vs RTX6000 逐条 route 对比（按 RouteScenario 编号）

4090 = 欠训练版（epoch12/bs1/~45k iter）；RTX6000 = 完整训练版（epoch60/bs16/225k iter）。
共同编号 214 条：变好 154、退步 25（其中 15 条为环境故障状态、非模型行为）、基本持平 35。

## 性能掉下来的编号（Δ<-3，共 25 条）

| RouteScenario ID | 场景 | 4090 DS | RTX6000 DS | Δ | RTX6000 状态 |
| ---: | --- | ---: | ---: | ---: | --- |
| 3436 | HazardAtSideLaneTwoWays_1 | 100.0 | 36.0 | -64.0 |  |
| 3749 | EnterActorFlow_1 | 70.0 | 15.1 | -54.9 |  |
| 24092 | MergerIntoSlowTrafficV2_1 | 100.0 | 60.0 | -40.0 |  |
| 25381 | HazardAtSideLane_1 | 100.0 | 60.0 | -40.0 |  |
| 2903 | NonSignalizedJunctionRightTurn_1 | 33.0 | 3.9 | -29.1 | 偏离路线 |
| 2403 | VanillaNonSignalizedTurn_1 | 26.4 | 0.0 | -26.4 | ⚠️环境:agent初始化失败 |
| 27532 | BlockedIntersection_1 | 33.4 | 7.5 | -25.9 | ⚠️环境:CARLA超时 |
| 27582 | PedestrianCrossing_1 | 24.1 | 0.9 | -23.2 |  |
| 2201 | EnterActorFlow_1 | 25.2 | 3.3 | -21.9 | ⚠️环境:CARLA超时 |
| 25975 | VanillaSignalizedTurnEncounterGreenLight_1 | 35.8 | 16.3 | -19.5 |  |
| 3410 | AccidentTwoWays_1 | 25.7 | 7.2 | -18.5 | ⚠️环境:CARLA超时 |
| 26966 | SignalizedJunctionRightTurn_1 | 25.3 | 8.6 | -16.7 | 卡住Blocked |
| 3890 | VanillaNonSignalizedTurn_1 | 25.5 | 10.9 | -14.6 |  |
| 25928 | VehicleOpensDoorTwoWays_1 | 36.0 | 21.6 | -14.4 |  |
| 3178 | VanillaNonSignalizedTurn_1 | 41.5 | 28.9 | -12.6 | 卡住Blocked |
| 27529 | PedestrianCrossing_1 | 25.4 | 13.3 | -12.1 | 卡住Blocked |
| 3676 | NonSignalizedJunctionRightTurn_1 | 21.7 | 10.1 | -11.6 | ⚠️环境:CARLA超时 |
| 2989 | BlockedIntersection_1 | 33.9 | 22.9 | -11.0 | 卡住Blocked |
| 3670 | NonSignalizedJunctionRightTurn_1 | 25.8 | 15.3 | -10.5 | ⚠️环境:CARLA超时 |
| 27018 | SignalizedJunctionRightTurn_1 | 24.8 | 14.6 | -10.2 | ⚠️环境:CARLA超时 |
| 3865 | VanillaSignalizedTurnEncounterRedLight_1 | 19.9 | 11.0 | -8.9 |  |
| 4937 | SignalizedJunctionRightTurn_1 | 12.5 | 4.0 | -8.4 | ⚠️环境:CARLA超时 |
| 2416 | VanillaNonSignalizedTurnEncounterStopsign_1 | 28.3 | 20.6 | -7.7 | 卡住Blocked |
| 25896 | ParkedObstacleTwoWays_1 | 9.0 | 2.2 | -6.9 | ⚠️环境:CARLA超时 |
| 28093 | NonSignalizedJunctionLeftTurnEnterFlow_1 | 25.6 | 19.0 | -6.6 |  |

## 性能变好的编号（Δ>+3，共 154 条，下表列全部）

| RouteScenario ID | 场景 | 4090 DS | RTX6000 DS | Δ |
| ---: | --- | ---: | ---: | ---: |
| 26393 | ParkingExit_1 | 2.7 | 100.0 | +97.3 |
| 26418 | ControlLoss_1 | 5.6 | 100.0 | +94.4 |
| 24785 | ConstructionObstacle_1 | 6.4 | 100.0 | +93.6 |
| 24258 | HazardAtSideLaneTwoWays_1 | 8.8 | 100.0 | +91.2 |
| 24330 | HardBreakRoute_1 | 9.4 | 100.0 | +90.6 |
| 1852 | AccidentTwoWays_1 | 10.0 | 100.0 | +90.0 |
| 2539 | Accident_1 | 10.4 | 100.0 | +89.6 |
| 26458 | T_Junction_1 | 0.0 | 88.4 | +88.4 |
| 4683 | SignalizedJunctionLeftTurn_1 | 13.5 | 100.0 | +86.5 |
| 2715 | StaticCutIn_1 | 13.6 | 100.0 | +86.4 |
| 23687 | HighwayExit_1 | 13.7 | 100.0 | +86.3 |
| 2844 | OppositeVehicleRunningRedLight_1 | 13.8 | 100.0 | +86.2 |
| 25383 | ParkedObstacle_1 | 16.7 | 100.0 | +83.3 |
| 2790 | InvadingTurn_1 | 17.0 | 100.0 | +83.0 |
| 28198 | SignalizedJunctionLeftTurnEnterFlow_1 | 17.8 | 100.0 | +82.2 |
| 3255 | ParkingCrossingPedestrian_1 | 18.0 | 100.0 | +82.0 |
| 10857 | VehicleTurningRoutePedestrian_1 | 18.9 | 100.0 | +81.1 |
| 2373 | VanillaSignalizedTurnEncounterRedLight_1 | 20.0 | 100.0 | +80.0 |
| 26405 | StaticCutIn_1 | 20.3 | 100.0 | +79.7 |
| 28111 | T_Junction_1 | 21.1 | 100.0 | +78.9 |
| 14842 | VanillaSignalizedTurnEncounterGreenLight_1 | 21.1 | 100.0 | +78.9 |
| 3561 | ControlLoss_1 | 21.6 | 100.0 | +78.4 |
| 3482 | VehicleOpensDoorTwoWays_1 | 21.6 | 100.0 | +78.4 |
| 2534 | Accident_1 | 24.1 | 100.0 | +75.8 |
| 24841 | ControlLoss_1 | 24.8 | 100.0 | +75.2 |
| 24078 | InterurbanAdvancedActorFlow_1 | 25.9 | 100.0 | +74.1 |
| 3093 | CrossingBicycleFlow_1 | 26.2 | 100.0 | +73.8 |
| 23930 | InterurbanActorFlow_1 | 0.0 | 73.4 | +73.4 |
| 23658 | HighwayExit_1 | 26.8 | 100.0 | +73.2 |
| 14909 | VanillaSignalizedTurnEncounterGreenLight_1 | 27.1 | 100.0 | +72.9 |
| 3712 | VehicleTurningRoute_1 | 8.5 | 80.0 | +71.5 |
| 23700 | InterurbanAdvancedActorFlow_1 | 28.7 | 100.0 | +71.3 |
| 25378 | YieldToEmergencyVehicle_1 | 0.0 | 70.0 | +70.0 |
| 24333 | DynamicObjectCrossing_1 | 30.0 | 100.0 | +70.0 |
| 17752 | DynamicObjectCrossing_1 | 30.0 | 100.0 | +70.0 |
| 2847 | OppositeVehicleRunningRedLight_1 | 30.1 | 100.0 | +69.9 |
| 23708 | InterurbanAdvancedActorFlow_1 | 33.6 | 100.0 | +66.4 |
| 2082 | OppositeVehicleRunningRedLight_1 | 35.6 | 100.0 | +64.4 |
| 3307 | Accident_1 | 36.0 | 100.0 | +64.0 |
| 2709 | StaticCutIn_1 | 36.0 | 100.0 | +64.0 |
| 25865 | ParkedObstacleTwoWays_1 | 36.0 | 100.0 | +64.0 |
| 18356 | ParkingCutIn_1 | 36.0 | 100.0 | +64.0 |
| 18311 | ParkingCutIn_1 | 36.0 | 100.0 | +64.0 |
| 18305 | ParkingCutIn_1 | 36.0 | 100.0 | +64.0 |
| 1773 | ParkedObstacle_1 | 36.0 | 100.0 | +64.0 |
| 23670 | HighwayExit_1 | 21.6 | 81.0 | +59.4 |
| 1956 | ParkingExit_1 | 1.9 | 60.0 | +58.1 |
| 3564 | InvadingTurn_1 | 42.4 | 100.0 | +57.6 |
| 2509 | ConstructionObstacle_1 | 8.3 | 65.0 | +56.7 |
| 3144 | VanillaSignalizedTurnEncounterRedLight_1 | 16.6 | 70.0 | +53.4 |
| 25300 | HazardAtSideLane_1 | 0.6 | 53.4 | +52.8 |
| 3869 | VanillaSignalizedTurnEncounterRedLight_1 | 18.0 | 70.0 | +52.0 |
| 25854 | HazardAtSideLaneTwoWays_1 | 8.5 | 60.0 | +51.5 |
| 24294 | ParkingCrossingPedestrian_1 | 50.0 | 100.0 | +50.0 |
| 3540 | HardBreakRoute_1 | 11.8 | 60.0 | +48.2 |
| 24759 | ParkingCutIn_1 | 14.3 | 60.0 | +45.7 |
| 23695 | InterurbanAdvancedActorFlow_1 | 55.0 | 100.0 | +45.0 |
| 3936 | SignalizedJunctionLeftTurn_1 | 15.1 | 60.0 | +44.9 |
| 2802 | InvadingTurn_1 | 15.5 | 60.0 | +44.5 |
| 24071 | InterurbanAdvancedActorFlow_1 | 56.5 | 100.0 | +43.5 |
| 26956 | SignalizedJunctionRightTurn_1 | 16.9 | 60.0 | +43.1 |
| 25845 | AccidentTwoWays_1 | 18.5 | 60.0 | +41.5 |
| 3575 | InvadingTurn_1 | 60.0 | 100.0 | +40.0 |
| 3572 | InvadingTurn_1 | 60.0 | 100.0 | +40.0 |
| 26401 | MergerIntoSlowTrafficV2_1 | 60.0 | 100.0 | +40.0 |
| 2554 | ParkedObstacle_1 | 60.0 | 100.0 | +40.0 |
| 2286 | HighwayCutIn_1 | 60.0 | 100.0 | +40.0 |
| 2643 | AccidentTwoWays_1 | 20.1 | 60.0 | +39.9 |
| 26396 | StaticCutIn_1 | 20.2 | 60.0 | +39.8 |
| 3464 | VehicleOpensDoorTwoWays_1 | 21.6 | 60.0 | +38.4 |
| 28219 | NonSignalizedJunctionLeftTurnEnterFlow_1 | 24.8 | 60.0 | +35.2 |
| 2129 | OppositeVehicleTakingPriority_1 | 11.9 | 46.5 | +34.7 |
| 3090 | CrossingBicycleFlow_1 | 2.8 | 37.4 | +34.5 |
| 2913 | OppositeVehicleTakingPriority_1 | 17.5 | 51.8 | +34.3 |
| 24240 | Accident_1 | 26.2 | 60.0 | +33.8 |
| 1711 | ParkingCutIn_1 | 21.6 | 55.0 | +33.4 |
| 2091 | NonSignalizedJunctionLeftTurn_1 | 15.7 | 48.0 | +32.3 |
| 3717 | VehicleTurningRoute_1 | 15.7 | 48.0 | +32.3 |
| 3905 | VanillaNonSignalizedTurnEncounterStopsign_1 | 19.6 | 51.8 | +32.2 |
| 28229 | NonSignalizedJunctionLeftTurnEnterFlow_1 | 9.8 | 42.0 | +32.2 |
| 14862 | VanillaSignalizedTurnEncounterGreenLight_1 | 27.9 | 60.0 | +32.1 |
| 24252 | DynamicObjectCrossing_1 | 18.0 | 50.0 | +32.0 |
| 28087 | NonSignalizedJunctionLeftTurnEnterFlow_1 | 17.0 | 48.0 | +31.0 |
| 27494 | BlockedIntersection_1 | 18.6 | 49.0 | +30.4 |
| 2668 | ParkedObstacleTwoWays_1 | 13.0 | 42.5 | +29.5 |
| 4468 | SignalizedJunctionLeftTurn_1 | 8.7 | 38.2 | +29.5 |
| 24781 | HardBreakRoute_1 | 7.8 | 36.0 | +28.2 |
| 24041 | HighwayExit_1 | 21.6 | 48.8 | +27.2 |
| 2144 | VehicleTurningRoute_1 | 12.1 | 37.9 | +25.8 |
| 2881 | NonSignalizedJunctionLeftTurn_1 | 11.6 | 36.0 | +24.4 |
| 27506 | EnterActorFlow_1 | 11.9 | 36.0 | +24.1 |
| 3048 | MergerIntoSlowTraffic_1 | 36.0 | 60.0 | +24.0 |
| 17635 | SequentialLaneChange_1 | 36.0 | 60.0 | +24.0 |
| 17598 | SequentialLaneChange_1 | 36.0 | 60.0 | +24.0 |
| 17569 | SequentialLaneChange_1 | 36.0 | 60.0 | +24.0 |
| 23901 | InterurbanActorFlow_1 | 21.6 | 45.6 | +24.0 |
| 26406 | HardBreakRoute_1 | 13.0 | 36.0 | +23.0 |
| 25439 | HazardAtSideLane_1 | 13.5 | 36.0 | +22.5 |
| 3708 | VehicleTurningRoute_1 | 15.4 | 37.2 | +21.8 |
| 3100 | CrossingBicycleFlow_1 | 15.3 | 37.0 | +21.7 |
| 3099 | CrossingBicycleFlow_1 | 13.4 | 35.0 | +21.7 |
| 24367 | ConstructionObstacle_1 | 1.1 | 22.7 | +21.6 |
| 24098 | InterurbanActorFlow_1 | 52.3 | 73.4 | +21.1 |
| 24784 | ControlLoss_1 | 30.3 | 51.1 | +20.8 |
| 3248 | ParkingCrossingPedestrian_1 | 30.0 | 50.0 | +20.0 |
| 24211 | DynamicObjectCrossing_1 | 30.0 | 50.0 | +20.0 |
| 18252 | ParkingCrossingPedestrian_1 | 30.0 | 50.0 | +20.0 |
| 14194 | PedestrianCrossing_1 | 80.0 | 100.0 | +20.0 |
| 2127 | OppositeVehicleTakingPriority_1 | 16.6 | 36.4 | +19.8 |
| 25424 | ConstructionObstacleTwoWays_1 | 6.8 | 26.3 | +19.5 |
| 28330 | SignalizedJunctionLeftTurnEnterFlow_1 | 15.1 | 34.3 | +19.3 |
| 25968 | VanillaSignalizedTurnEncounterGreenLight_1 | 24.1 | 43.3 | +19.2 |
| 23910 | InterurbanActorFlow_1 | 52.3 | 71.4 | +19.1 |
| 2164 | VehicleTurningRoutePedestrian_1 | 1.9 | 20.4 | +18.5 |
| 26944 | OppositeVehicleRunningRedLight_1 | 0.0 | 18.4 | +18.4 |
| 28035 | T_Junction_1 | 5.0 | 23.1 | +18.1 |
| 1833 | ConstructionObstacleTwoWays_1 | 16.7 | 34.6 | +17.9 |
| 2943 | VehicleTurningRoute_1 | 6.7 | 23.3 | +16.5 |
| 28241 | SignalizedJunctionLeftTurnEnterFlow_1 | 13.0 | 29.1 | +16.2 |
| 3086 | CrossingBicycleFlow_1 | 26.1 | 41.5 | +15.4 |
| 24206 | ParkingCrossingPedestrian_1 | 14.8 | 30.0 | +15.2 |
| 3189 | VanillaNonSignalizedTurnEncounterStopsign_1 | 20.4 | 35.2 | +14.8 |
| 24340 | ControlLoss_1 | 29.2 | 43.8 | +14.6 |
| 25955 | HazardAtSideLaneTwoWays_1 | 21.8 | 36.0 | +14.2 |
| 26990 | NonSignalizedJunctionLeftTurn_1 | 1.6 | 14.9 | +13.3 |
| 3457 | ParkedObstacleTwoWays_1 | 22.7 | 36.0 | +13.3 |
| 25358 | StaticCutIn_1 | 0.0 | 13.0 | +13.0 |
| 28099 | SignalizedJunctionLeftTurnEnterFlow_1 | 22.8 | 35.2 | +12.5 |
| 26950 | OppositeVehicleRunningRedLight_1 | 21.3 | 33.3 | +12.1 |
| 1825 | ConstructionObstacleTwoWays_1 | 11.7 | 23.7 | +12.0 |
| 3666 | NonSignalizedJunctionRightTurn_1 | 17.9 | 29.9 | +11.9 |
| 2664 | ParkedObstacleTwoWays_1 | 24.4 | 36.1 | +11.7 |
| 2050 | SignalizedJunctionRightTurn_1 | 24.8 | 36.3 | +11.5 |
| 25951 | HazardAtSideLaneTwoWays_1 | 7.6 | 19.1 | +11.5 |
| 20920 | ConstructionObstacleTwoWays_1 | 11.6 | 22.2 | +10.6 |
| 28243 | SignalizedJunctionLeftTurnEnterFlow_1 | 13.8 | 24.3 | +10.6 |
| 17655 | SequentialLaneChange_1 | 49.5 | 60.0 | +10.5 |
| 4183 | SignalizedJunctionLeftTurn_1 | 21.8 | 32.0 | +10.1 |
| 2084 | NonSignalizedJunctionLeftTurn_1 | 10.3 | 19.6 | +9.2 |
| 2204 | BlockedIntersection_1 | 27.5 | 36.7 | +9.2 |
| 2513 | ConstructionObstacle_1 | 5.8 | 14.6 | +8.7 |
| 3184 | VanillaNonSignalizedTurnEncounterStopsign_1 | 20.4 | 29.0 | +8.6 |
| 2086 | NonSignalizedJunctionLeftTurn_1 | 10.3 | 18.8 | +8.6 |
| 28048 | T_Junction_1 | 5.7 | 14.2 | +8.5 |
| 2273 | MergerIntoSlowTraffic_1 | 91.9 | 100.0 | +8.1 |
| 2606 | ConstructionObstacleTwoWays_1 | 26.6 | 33.4 | +6.7 |
| 28154 | T_Junction_1 | 13.0 | 19.4 | +6.5 |
| 4669 | SignalizedJunctionLeftTurn_1 | 17.7 | 23.9 | +6.2 |
| 3731 | VehicleTurningRoutePedestrian_1 | 2.4 | 8.2 | +5.8 |
| 2143 | OppositeVehicleTakingPriority_1 | 22.4 | 28.0 | +5.6 |
| 3876 | VanillaSignalizedTurnEncounterRedLight_1 | 12.5 | 17.9 | +5.4 |
| 27515 | PedestrianCrossing_1 | 22.8 | 28.1 | +5.2 |
| 2283 | MergerIntoSlowTraffic_1 | 55.7 | 60.0 | +4.3 |
| 2397 | VanillaNonSignalizedTurn_1 | 25.3 | 28.8 | +3.5 |
