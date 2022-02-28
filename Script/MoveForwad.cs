using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;

public class MoveForwad : Agent
{
    private Transform Body; //crawing robot transform
    public Transform Target; // target transform

    [Header("Move Parts")] [Space(10)] public Transform body;
    public Transform Arm1;
    public Transform Arm2;
    JointDriveController m_JdController;
    
    [HideInInspector]
    public Vector3 distanceToTarget;
    public int decisionCounter;
    public override void Initialize(){
        Body = GetComponent<Transform>();

        m_JdController = GetComponent<JointDriveController>();

        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(Arm1);
        m_JdController.SetupBodyPart(Arm2);
    }
    // When the episode begins, perform
    public override void OnEpisodeBegin(){

        transform.localPosition = new Vector3(0, 14f, 0);

        Target.localPosition = new Vector3(Random.Range(-200f, -80f), 5f, 0); // X coordinate value: -80 ~ -200

        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }
    }

    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }

    }
    // A method of observing and collecting environmental information and delivering it to the reinforcement trainer for policy making.
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.position);
        sensor.AddObservation(m_JdController.bodyPartsDict[body].rb.position);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }
    // A method of executing actions received from the policy
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var bpDict = m_JdController.bodyPartsDict;

        var continuousActions = actionBuffers.ContinuousActions;
        var i = -1;
        
        // Pick a new target joint rotation
        bpDict[Arm1].SetJointTargetRotation(0, 0, continuousActions[++i]);
        bpDict[Arm2].SetJointTargetRotation(0, 0, continuousActions[++i]);

        // Update joint strength
        //bpDict[Arm1].SetJointStrength(continuousActions[++i]);
        //bpDict[Arm2].SetJointStrength(continuousActions[++i]);
    }

    private void FixedUpdate(){
        distanceToTarget = Target.position - m_JdController.bodyPartsDict[body].rb.position;

        if(decisionCounter == 0){
            decisionCounter = 3;
            RequestDecision();
        }
        else{
            decisionCounter--;
        }

        RewardFunctionMovingTowards();
        RewardFunctionTimePenalty();
    }

    void RewardFunctionMovingTowards(){
        float movingForwardDot = Vector3.Dot(m_JdController.bodyPartsDict[body].rb.velocity, distanceToTarget.normalized);
        AddReward(0.01f * movingForwardDot);
    }

    void RewardFunctionTimePenalty(){
        AddReward(-0.001f); //-0.001f chosen by experimentation. 
    }

}
